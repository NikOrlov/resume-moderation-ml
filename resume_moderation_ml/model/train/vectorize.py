import numpy as np
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from resume_moderation_ml.model import text_fields
from resume_moderation_ml.model.resume import (
    POSSIBLE_PHONE_FIELDS_EXTRACTOR,
    FieldSizeQuotients,
    ManualRejectRules,
    ResumeNullFieldsCount,
    ResumeNumberStats,
    SalaryExtractor,
    StartOfCareerConsistency,
    get_age,
    get_age_experience_difference,
    get_gender,
    get_industry_fullness,
    get_known_university_count,
    get_longest_phone_char,
    get_time_without_education,
    get_time_without_work,
    get_title_delimiters_count,
    get_total_experience,
    has_photo,
    is_incomplete,
)
from resume_moderation_ml.model.train import bad_positions, cache_obj, source
from resume_moderation_ml.model.train.logger import setup_logger
from resume_moderation_ml.model.train.utils import alpha_quotient, cache, caps_quotient, load_currency_rates
from resume_moderation_ml.model.train.utils.text import Analyzer, OutOfVocabularyTokens, TokenLengthStatistics
from resume_moderation_ml.model.train.utils.transformers import (
    CategoricalEncoder,
    ClassifierTransformer,
    InputLength,
    LogTransformer,
    ValueExtractor,
    make_field_length_extractor,
)
from resume_moderation_ml.model.train.vocabulary import get_vocabularies

logger = setup_logger(__name__)

_RESUME_VECTORS_KEY = "vectorizer/resume_vectors"
_FEATURE_NAMES_KEY = "vectorizer/feature_names"
_VECTORIZER_NAMES_KEY = "vectorizer/vectorizer"


@cache.cache(_VECTORIZER_NAMES_KEY, cache_cls=cache_obj)
def get_vectorizer_pipe():
    # start fitting vectorizer
    resumes = source.get_vectorizer_resumes()
    logger.info("loaded %d resumes to fit vectorizer", len(resumes))
    transformers = []
    vocabularies = get_vocabularies(text_fields.FIELDS_TO_ANALYZE)

    feature_names = []

    # manual rejection rules
    transformers.append(("manual_rejection", ValueExtractor(ManualRejectRules())))
    feature_names.append("manual_rejection")

    # complete tf-idf features, word count and out of vocab features for main text fields
    for field in text_fields.FIELDS_TO_ANALYZE:
        vocabulary = vocabularies[field]
        vectorizer = text_fields.create_vectorizer(field, vocabulary)

        field_pipe = Pipeline(
            [
                ("extractor", text_fields.create_extractor(field)),
                ("analyzer", text_fields.create_analyzer(field)),
                (
                    "text_features",
                    FeatureUnion(
                        [
                            ("tf_idf", vectorizer),
                            ("word_count", InputLength()),
                            ("out_of_vocab", OutOfVocabularyTokens(vocabulary, vectorizer.stop_words)),
                        ]
                    ),
                ),
            ]
        ).fit(resumes)

        transformers.append((field, field_pipe))
        feature_names.extend(
            [
                "{}__{}".format(field, feature_name)
                for feature_name in field_pipe.named_steps["text_features"].get_feature_names()
            ]
        )

    # features, derived from full text
    full_text_pipe = Pipeline(
        [
            ("extract", text_fields.create_extractor("full_text")),
            (
                "union",
                FeatureUnion(
                    [
                        (
                            "token_features",
                            Pipeline(
                                [
                                    ("analyzer", Analyzer(use_stemming=False)),
                                    (
                                        "union",
                                        FeatureUnion(
                                            [
                                                (
                                                    "statistics",
                                                    TokenLengthStatistics(percentiles=np.linspace(0.0, 100.0, 11)),
                                                ),
                                                ("word_count", InputLength()),
                                            ]
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "char_features",
                            FeatureUnion(
                                [
                                    ("caps_quotient", ValueExtractor(caps_quotient)),
                                    ("alpha_quotient", ValueExtractor(alpha_quotient)),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )
    transformers.append(("full_text_features", full_text_pipe))
    feature_names.extend(
        [
            "full_text__{}".format(feature_name)
            for feature_name in full_text_pipe.named_steps["union"]
            .transformer_list[0][1]
            .named_steps["union"]
            .get_feature_names()
        ]
    )
    feature_names.extend(("caps_quotient", "alpha_quotient"))

    # bad positions
    transformers.append(("bad_positions", ClassifierTransformer(bad_positions.get_model())))
    feature_names.append("bad_positions_prob")

    # field size quotients
    field_size_quotients = FieldSizeQuotients()
    transformers.append(("field_size_quotients", ValueExtractor(field_size_quotients)))
    feature_names.extend(field_size_quotients.feature_names)

    # number sequences stats (for detecting possible phone numbers in text)
    resume_number_stats = ResumeNumberStats()
    transformers.append(
        (
            "number_sequences_stats",
            Pipeline([("extract", POSSIBLE_PHONE_FIELDS_EXTRACTOR), ("stats", ValueExtractor(resume_number_stats))]),
        )
    )
    feature_names.extend(resume_number_stats.feature_names)

    # count of resume empty fields
    transformers.append(("nulls", ResumeNullFieldsCount()))
    feature_names.append("null_fields_count")

    # count of delimiters in title
    transformers.append(("title_delimiters", ValueExtractor(get_title_delimiters_count, dtype=np.float64)))
    feature_names.append("title_delimiters")

    # times without work and education
    transformers.append(("time_without_work", ValueExtractor(get_time_without_work)))
    feature_names.append("log_time_without_work")
    transformers.append(("time_without_education", ValueExtractor(get_time_without_education)))
    feature_names.append("log_time_without_education")

    # salary, total experience, age, gender, photo
    transformers.append(
        (
            "salary",
            Pipeline([("salary", ValueExtractor(SalaryExtractor(load_currency_rates()))), ("log", LogTransformer())]),
        )
    )
    feature_names.append("log_salary")

    transformers.append(
        (
            "total_experience",
            Pipeline([("total_experience", ValueExtractor(get_total_experience)), ("log", LogTransformer())]),
        )
    )
    feature_names.append("log_total_experience")
    transformers.append(("age", ValueExtractor(get_age, dtype=np.float64)))
    feature_names.append("age")

    gender_pipe = Pipeline(
        [
            ("extract", ValueExtractor(get_gender)),
            ("encode", CategoricalEncoder()),
            ("one_hot", OneHotEncoder(sparse=False)),
        ]
    ).fit(resumes)
    transformers.append(("gender", gender_pipe))
    feature_names.extend("gender_{}".format(value) for value in gender_pipe.named_steps["encode"].values_[:-1])

    transformers.append(("has_photo", ValueExtractor(has_photo, dtype=np.float64)))
    feature_names.append("has_photo")
    transformers.append(("age_experience_difference", ValueExtractor(get_age_experience_difference, dtype=np.float64)))
    feature_names.append("age_experience_difference")

    # field lengths
    transformers.extend(
        (
            ("experience_count", make_field_length_extractor("experience")),
            ("primary_education_count", make_field_length_extractor("primaryEducation")),
            ("additional_education_count", make_field_length_extractor("additionalEducation")),
            ("attestation_education_count", make_field_length_extractor("attestationEducation")),
            ("key_skills_count", make_field_length_extractor("keySkills")),
            ("first_name_length", make_field_length_extractor("firstName")),
            ("middle_name_length", make_field_length_extractor("middleName")),
            ("last_name_length", make_field_length_extractor("lastName")),
        )
    )
    feature_names.extend(
        [
            "{}_len".format(f)
            for f in (
                "experience",
                "primaryEducation",
                "additionalEducation",
                "attestationEducation",
                "keySkills",
                "firstName",
                "middleName",
                "lastName",
            )
        ]
    )

    # start of career consistency
    start_of_career_consistency = StartOfCareerConsistency()
    transformers.append(("start_of_career_consistency", ValueExtractor(start_of_career_consistency)))
    feature_names.extend(start_of_career_consistency.feature_names)

    # longest phone one-char sequence
    transformers.append(("phone_onechar_sequence", ValueExtractor(get_longest_phone_char, dtype=np.float64)))
    feature_names.append("phone_onechar_sequence")

    # industry and university fullness
    transformers.append(("industry_fullness", ValueExtractor(get_industry_fullness)))
    feature_names.append("industry_fullness")
    transformers.append(("known_university_count", ValueExtractor(get_known_university_count, dtype=np.float64)))
    feature_names.append("known_university_count")

    transformers.append(("incomplete", ValueExtractor(is_incomplete)))
    feature_names.append("incomplete")

    vectorizer_pipe = FeatureUnion(transformers)
    logger.info("got %d features", len(feature_names))
    cache_obj.save(feature_names, _FEATURE_NAMES_KEY)

    logger.info("started fitting source resume vectorizer with %d resumes", len(resumes))
    vectorizer_pipe.fit(resumes)
    logger.info("finished fitting source resume vectorizer")
    return vectorizer_pipe


@cache.cache(_RESUME_VECTORS_KEY, cache_cls=cache_obj)
def get_resume_vectors():
    vectorizer = get_vectorizer_pipe()
    resumes = source.get_raw_resumes()
    logger.info("transform %d source resumes to a vector representation", len(resumes))
    resume_vectors = vectorizer.transform(resumes)
    logger.info("shape of transformed data is %s", repr(resume_vectors.shape))
    return resume_vectors


def get_feature_names():
    return cache_obj.load(_FEATURE_NAMES_KEY)


if __name__ == "__main__":
    get_resume_vectors()
