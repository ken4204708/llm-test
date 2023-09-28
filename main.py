#
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# DO NOT EDIT! This is a generated sample ("Request",  "language_classify_text")

# To install the latest published package dependency, execute the following:
#   pip install google-cloud-language

# sample-metadata
#   title: Classify Content
#   description: Classifying Content in a String
#   usage: python3 samples/v1/language_classify_text.py [--text_content "That actor on TV makes movies in Hollywood and also stars in a variety of popular new TV shows."]

# [START language_classify_text]
from google.cloud import language_v1
import pandas as pd
import numpy as np
import argparse
import json
import time
import os

# Segmentation package
import pkg_resources
from symspellpy.symspellpy import SymSpell

# from ekphrasis.classes.segmenter import Segmenter
from google.oauth2 import service_account

service_account_key_path = 'polar-arbor-250703-444efe112ad6.json'
credentials = service_account.Credentials.from_service_account_file(
    service_account_key_path,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)
def sample_classify_text(text_content):
    """
    Classifying Content in a String

    Args:
      text_content The text content to analyze.
    """

    client = language_v1.LanguageServiceClient(credentials=credentials)

    # text_content = "That actor on TV makes movies in Hollywood and also stars in a variety of popular new TV shows."

    # Available types: PLAIN_TEXT, HTML
    type_ = language_v1.Document.Type.PLAIN_TEXT

    # Optional. If not specified, the language is automatically detected.
    # For list of supported languages:
    # https://cloud.google.com/natural-language/docs/languages
    language = "en"
    document = {"content": text_content, "type_": type_, "language": language}

    content_categories_version = (
        language_v1.ClassificationModelOptions.V2Model.ContentCategoriesVersion.V2
    )
    response = client.classify_text(
        request={
            "document": document,
            "classification_model_options": {
                "v2_model": {"content_categories_version": content_categories_version}
            },
        }
    )
    # Loop through classified categories returned from the API
    # for category in response.categories:
        # Get the name of the category representing the document.
        # See the predefined taxonomy of categories:
        # https://cloud.google.com/natural-language/docs/categories
        # print(f"Category name: {category.name}")
        # Get the confidence. Number representing how certain the classifier
        # is that this category represents the provided text.
        # print(f"Confidence: {category.confidence}")
    if response:
        confidences = [x.confidence for x in response.categories]
        category_with_highest_confidence = response.categories[np.argmax(confidences)].name
    else:
        confidences = [0]
        category_with_highest_confidence = ''
    category_with_highest_confidence = category_with_highest_confidence.split('/')
    length_category = len(category_with_highest_confidence)
    for i in np.arange(length_category,4):
        category_with_highest_confidence.append('')
    return max(confidences), category_with_highest_confidence[1], category_with_highest_confidence[2], category_with_highest_confidence[3]
    # return max(confidences), category_with_highest_confidence.replace('/', '|')

# [END language_classify_text]

def read_csv_file(file_name, dropna_rule):
    raw_df = pd.read_csv(file_name)
    print(f'''There are {len(raw_df)} rows in file {file_name}''')
    # raw_df = raw_df.dropna(subset = dropna_rule)
    keep_index = raw_df.dropna(subset = dropna_rule).index
    raw_df['keep_index'] = raw_df.index.isin(keep_index)
    return raw_df

def f1(row, select_cols, expr, model):
    print(row['Name'])
    if row['keep_index']:
        select_results = row[select_cols].dropna().to_list()
        row['combined_name'] = text_1 = expr.join(select_results)
        seg_text = model.word_segmentation(row['combined_name'])
        row['seg_combined_name'] = text_2 = seg_text.corrected_string
        row['orginal_conf'], row['original_classify_level_1'], row['original_classify_level_2'], row['original_classify_level_3'] = sample_classify_text(text_1)
        row['seg_conf'], row['seg_classify_level_1'], row['seg_classify_level_2'], row['seg_classify_level_3'] = sample_classify_text(text_2)
        row['classify_result_based_high_conf_level_1'] = row['original_classify_level_1'] if row['orginal_conf'] >= \
                                                            row['seg_conf'] else row['seg_classify_level_1']
        row['classify_result_based_high_conf_level_2'] = row['original_classify_level_2'] if row['orginal_conf'] >= \
                                                    row['seg_conf'] else row['seg_classify_level_2']
        row['classify_result_based_high_conf_level_3'] = row['original_classify_level_3'] if row['orginal_conf'] >= \
                                                    row['seg_conf'] else row['seg_classify_level_3']
        time.sleep(0.3)
    return row


def main():
    with open(args['json'], 'r') as f:
        files = json.load(f)

    sym_spell = SymSpell(max_dictionary_edit_distance=0, prefix_length=7)
    dictionary_path = pkg_resources.resource_filename(
        "symspellpy", "frequency_dictionary_en_82_765.txt"
    )
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)

    for file in files:
        file_name = file['filename']
        fn, ext = os.path.splitext(file_name)
        drop_cols, sel_cols, expr = file['drop_cols'], file['sel_cols'], file['expr']
        raw_df = read_csv_file(file_name, drop_cols)
        raw_df = raw_df.apply(
            f1,
            select_cols=sel_cols,
            expr=expr,
            model=sym_spell,
            axis=1
        )

        # raw_df = raw_df.drop(columns=['classify_result'])
        result_df = pd.DataFrame([])
        result_df["CustomerCode"] = raw_df["CustomerCode"]
        result_df["AutoID"] = raw_df["AutoID"]
        result_df[sel_cols] = raw_df[sel_cols]
        result_df["combined_name"] = raw_df["combined_name"]
        result_df['seg_combined_name'] = raw_df['seg_combined_name']
        result_df["classify_result_based_high_conf_level_1"] = raw_df["classify_result_based_high_conf_level_1"]
        result_df["classify_result_based_high_conf_level_2"] = raw_df["classify_result_based_high_conf_level_2"]
        result_df["classify_result_based_high_conf_level_3"] = raw_df["classify_result_based_high_conf_level_3"]
        result_df['orginal_conf'] = raw_df['orginal_conf']
        result_df["original_classify_level_1"] = raw_df["original_classify_level_1"]
        result_df["original_classify_level_2"] = raw_df["original_classify_level_2"]
        result_df["original_classify_level_3"] = raw_df["original_classify_level_3"]
        result_df['seg_conf'] = raw_df['seg_conf']
        result_df["seg_classify_level_1"] = raw_df["seg_classify_level_1"]
        result_df["seg_classify_level_2"] = raw_df["seg_classify_level_2"]
        result_df["seg_classify_level_3"] = raw_df["seg_classify_level_3"]
        result_df["classify_result_based_high_conf_level_1"] = raw_df["classify_result_based_high_conf_level_1"]
        # result_df = raw_df
        result_df = result_df.sort_values(by=["classify_result_based_high_conf_level_1", "classify_result_based_high_conf_level_2"])
        result_df.to_csv(f"output/{fn}_output{ext}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="JSON file", default="config.json")
    args, _ = parser.parse_known_args()
    args = vars(args)

    main()
