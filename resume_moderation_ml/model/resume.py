import re
from itertools import chain

import numpy as np
from sklearn.base import TransformerMixin

from resume_moderation_ml.model.stopwords import stop_positions
from resume_moderation_ml.model.text_fields import create_extractor
from resume_moderation_ml.model.train.utils import timestamp_from_string, timestamp_from_year
from resume_moderation_ml.model.train.utils.text import Analyzer
from resume_moderation_ml.model.train.utils.transformers import JsonTextExtractor, NoFitMixin

_ALL_RESUME_FIELDS = [
    "accessType",
    "additionalEducation",
    "advancedKeySkills",
    "age",
    "area",
    "attestationEducation",
    "birthday",
    "blacklist",
    "businessTripReadiness",
    "certificate",
    "citizenship",
    "creationTime",
    "educationLevel",
    "elementaryEducation",
    "email",
    "employment",
    "firstName",
    "gender",
    "id",
    "keySkills",
    "lang",
    "language",
    "lastChangeTime",
    "lastChangeTimeDetails",
    "lastExperience",
    "lastName",
    "markService",
    "metro",
    "metroStation",
    "middleName",
    "moderFullName",
    "moderationFlags",
    "moderationNotes",
    "moderationTime",
    "personalSite",
    "phone",
    "photo",
    "photoUrls",
    "portfolio",
    "portfolioUrls",
    "preferredContact",
    "primaryEducation",
    "recommendation",
    "relocation",
    "relocationArea",
    "renewalService",
    "salary",
    "secretWord",
    "skills",
    "source",
    "status",
    "title",
    "totalExperience",
    "travelTime",
    "userId",
    "vacancyTypesForResponse",
    "whitelist",
    "workSchedule",
    "workTicket",
]

_NON_INFORMATIONAL_FIELDS = [
    "accessType",
    "blacklist",
    "id",
    "lang",
    "lastChangeTime",
    "lastChangeTimeDetails",
    "lastExperience",
    "markService",
    "metro",
    "moderFullName",
    "moderationFlags",
    "moderationNotes",
    "moderationTime",
    "photoUrls",
    "portfolioUrls",
    "relocation",
    "renewalService",
    "secretWord",
    "source",
    "status",
    "userId",
    "vacancyTypesForResponse",
    "whitelist",
]

_INFORMATIONAL_FIELDS = [field for field in _ALL_RESUME_FIELDS if field not in _NON_INFORMATIONAL_FIELDS]


def _count_null_fields(resume):
    return sum(1 for field in _INFORMATIONAL_FIELDS if resume.get(field, None) is None)


class ResumeNullFieldsCount(TransformerMixin, NoFitMixin):
    def transform(self, X):
        return np.array(list(map(_count_null_fields, X)), dtype=np.float64).reshape((len(X), 1))


def get_title_delimiters_count(resume):
    if not resume.get("title"):
        return 0
    return sum(1 for ch in resume["title"] if ch in ".,;/")


def get_time_without_work(resume):
    if "lastExperience" in resume:
        last_experience = resume["lastExperience"]
    else:
        experiences = resume["experience"]
        last_experience = experiences[0] if experiences else None

    if not last_experience:
        return -1.0

    work_end_date = last_experience["endDate"]
    diff = 0.0
    if work_end_date:
        try:
            diff = max(0.0, resume["lastChangeTime"] - timestamp_from_string(work_end_date))
        except Exception:
            pass

    return np.log(1.0 + diff)


# TODO: maybe only primary education should be taken into account?
def get_time_without_education(resume):
    educations = [e for e in resume["primaryEducation"] + resume["additionalEducation"] if e["year"] is not None]
    if not educations:
        return -1.0

    studying_end_date = max(int(e["year"]) for e in educations)

    diff = resume["lastChangeTime"] - timestamp_from_year(studying_end_date)
    if diff < 0:
        return 0.0
    return np.log(1.0 + diff)


def extract_phone_number(phone_json):
    return "".join(
        re.sub(r"[^0-9]*", "", phone_json[part])
        for part in ("country", "city", "number")
        if phone_json[part] is not None
    )


def get_longest_phone_char(resume):
    if not resume["phone"]:
        return 0

    result = 0

    for phone in resume["phone"]:
        previous = None
        current_length = 0

        for char in extract_phone_number(phone):
            if char == previous:
                current_length += 1
            else:
                result = max(result, current_length)
                current_length = 1
            previous = char

        result = max(result, current_length)

    return result


def longest_non_zero_seq(string):
    longest = 0
    current = 0
    for ch in string:
        if ch != "0":
            current += 1
        else:
            longest = max(current, longest)
            current = 0
    longest = max(current, longest)
    return longest


class SalaryExtractor(object):
    def __init__(self, currency_rates):
        self.currency_rates = currency_rates

    def __call__(self, resume):
        if not resume["salary"]:
            return 0.0

        salary = resume["salary"]
        # return float(self.currency_rates.get(salary["currency"], 1.0) * salary["amount"])
        return float(salary["amount"] / self.currency_rates.get(salary["currency"], 1.0))


class SalaryDropFeaturesExtractor:
    def __init__(self, currency_rates):
        self.salary_extractor = SalaryExtractor(currency_rates)

    def __call__(self, resume):
        if not resume["salary"]:
            raise ValueError()
        amount = resume["salary"]["amount"]
        amount_str = str(amount)

        amount_len = len(amount_str)
        zero_count = amount_str.count("0")
        zero_quotient = zero_count / amount_len
        longest_nzs = longest_non_zero_seq(amount_str)
        longest_nzs_quotient = longest_nzs / amount_len

        return np.array(
            [
                amount,
                self.salary_extractor(resume),
                amount_len,
                zero_count,
                zero_quotient,
                longest_nzs,
                longest_nzs_quotient,
            ]
        )


def get_platform(resume):
    return resume["platform"]


def get_total_experience(resume):
    return resume["totalExperience"]


def get_age(resume):
    return resume["age"] if resume["age"] is not None else -1


def get_gender(resume):
    return resume["gender"]


def get_age_experience_difference(resume):
    if resume["age"] is None:
        return -9999.0

    return resume["age"] - resume["totalExperience"] / 12


class StartOfCareerConsistency(object):
    def __init__(self):
        self.feature_names = [
            "adult_and_inexperienced",
        ]

    def __call__(self, resume):
        adult = get_age(resume) >= 27
        inexperienced = get_total_experience(resume) < 12  # total experience is in months

        return np.array(
            [
                adult and inexperienced,
            ],
            dtype=float,
        )


def has_photo(resume):
    return resume["photo"] is not None


def is_incomplete(resume):
    return "validationSchema" in resume and resume["validationSchema"] == "incomplete"


def get_industry_fullness(resume):
    experiences = resume["experience"]
    if not experiences:
        return -1.0
    return sum(1.0 for exp in experiences if exp["companyIndustryId"] is not None) / len(experiences)


def get_known_university_count(resume):
    educations = resume["primaryEducation"]
    if not educations:
        return -1
    return sum(1 for edu in educations if edu["universityId"] is not None)


def _make_index_exp_query(index=None):
    if index is not None:
        return {"experience": {index: ["companyName", "description", "position"]}}
    else:
        return {"experience": ["companyName", "description", "position"]}


_EXP_LEN_PERCENTILES = np.linspace(0.0, 100.0, 11)


def get_token_count(text, analyzer):
    return float(len(analyzer.analyze(text)))


class FieldSizeQuotients(object):
    def __init__(self):
        feature_names = [
            "last_exp_len",
            "last_exp_to_full_exp",
            "last_exp_to_full_text",
            "full_exp_to_full_text",
            "full_edu_to_full_text",
        ]
        for prefix in ("exp_len", "exp_len_to_full_exp", "exp_len_to_full_text"):
            feature_names.extend(["{}_perc_{:.1f}".format(prefix, perc) for perc in _EXP_LEN_PERCENTILES])
            feature_names.extend(["{}_{}".format(prefix, name) for name in ("mean", "std")])
        self.feature_names = feature_names

    def __call__(self, resume):
        analyzer = Analyzer(use_stemming=False)

        full_exp_len = get_token_count(create_extractor("experience").fit_transform([resume])[0], analyzer)
        last_exp_len = get_token_count(JsonTextExtractor(_make_index_exp_query(0)).fit_transform([resume])[0], analyzer)
        full_edu_len = get_token_count(create_extractor("education").fit_transform([resume])[0], analyzer)
        full_text_len = get_token_count(create_extractor("full_text").fit_transform([resume])[0], analyzer)

        exp_len_percentiles = np.zeros(_EXP_LEN_PERCENTILES.size + 2)
        if resume["experience"]:
            exp_lens = []
            for i in range(len(resume["experience"])):
                exp_lens.append(
                    len(analyzer.analyze(JsonTextExtractor(_make_index_exp_query(i)).fit_transform([resume])[0]))
                )
            exp_lens = np.array(exp_lens)
            exp_len_percentiles = np.concatenate(
                [
                    np.array([np.percentile(exp_lens, p) for p in _EXP_LEN_PERCENTILES]),
                    np.array([exp_lens.mean(), exp_lens.std()]),
                ]
            )

        stats = [
            np.array(
                [
                    last_exp_len,
                    last_exp_len / (1.0 + full_exp_len),
                    last_exp_len / (1.0 + full_text_len),
                    full_exp_len / (1.0 + full_text_len),
                    full_edu_len / (1.0 + full_text_len),
                ]
            )
        ]
        stats.extend(
            [
                exp_len_percentiles,
                exp_len_percentiles / (1.0 + full_exp_len),
                exp_len_percentiles / (1.0 + full_text_len),
            ]
        )
        return np.concatenate(stats)


def get_number_sequences_stats(numbers):
    if not numbers:
        return np.zeros(11)

    lengths = np.array(list(map(len, numbers)))
    stats = [
        (lengths >= 7).sum(),
        len([num for num in numbers if num[0] in "78" and len(num) >= 7]),
        lengths.size,
        lengths.mean(),
        lengths.std(),
    ]
    stats.extend(np.percentile(lengths, p) for p in np.linspace(0.0, 100.0, 6))
    return np.array(stats)


_NUM_PATTERN = re.compile(r"\d+")
_PHONE_NUMBER_SUSPICION = re.compile(r"(?u)[-\d\s+()]*\d+[-\d\s+()]*")
POSSIBLE_PHONE_FIELDS_EXTRACTOR = JsonTextExtractor(
    {
        "experience": ["companyName", "description", "position"],
        "skills": None,
        "recommendation": ["organization", "position"],
    }
)


class ResumeNumberStats(object):
    def __init__(self):
        feature_names = []
        for prefix in ("num", "phone"):
            feature_names.extend(
                [
                    "{}_{}".format(prefix, feature_name)
                    for feature_name in ("long_count", "long_78_count", "count", "len_mean", "len_std")
                ]
            )
            feature_names.extend(["{}_len_perc_{:.1f}".format(prefix, perc) for perc in np.linspace(0.0, 100.0, 6)])
        self.feature_names = feature_names

    def __call__(self, text):
        return np.concatenate(
            [
                get_number_sequences_stats(list(filter(None, _NUM_PATTERN.findall(text)))),
                get_number_sequences_stats(
                    list(
                        filter(
                            None,
                            ["".join(_NUM_PATTERN.findall(item)) for item in _PHONE_NUMBER_SUSPICION.findall(text)],
                        )
                    )
                ),
            ]
        )


class ManualRejectRules(object):
    def __call__(self, resume):
        if resume["portfolio"]:
            return 1.0

        if "validationSchema" in resume and resume["validationSchema"] == "incomplete":
            return 0.0

        if (resume["age"] is None or resume["age"] >= 27) and not resume["experience"]:
            return 1.0

        if not resume["title"] or resume["title"].strip().lower() in stop_positions:
            return 1.0

        fields = POSSIBLE_PHONE_FIELDS_EXTRACTOR.transform([resume])[0]
        for token in chain.from_iterable(_PHONE_NUMBER_SUSPICION.findall(text) for text in fields):
            if len("".join(_NUM_PATTERN.findall(token))) >= 9:
                return 1.0

        return 0.0
