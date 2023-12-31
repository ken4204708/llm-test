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

from ekphrasis.classes.segmenter import Segmenter

def sample_classify_text(text_content):
    """
    Classifying Content in a String

    Args:
      text_content The text content to analyze.
    """

    client = language_v1.LanguageServiceClient()

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
    return max(confidences), category_with_highest_confidence

# [END language_classify_text]

def read_csv_file(file_name, dropna_rule):
    raw_df = pd.read_csv(file_name)
    print(f'''There are {len(raw_df)} rows in file {file_name}''')
    keep_index = raw_df.dropna(subset = dropna_rule).index
    raw_df['keep_index'] = raw_df.index.isin(keep_index)
    return raw_df

def f1(row, select_cols, expr, model):
    if row['keep_index']:
        select_results = row[select_cols].dropna().to_list()
        final_text = text = expr.join(select_results)
        seg_text = model.word_segmentation(text)
        final_text = seg_text.corrected_string
        conf_1, res_1 = sample_classify_text(text)
        conf_2, res_2 =sample_classify_text(final_text)
        final_result = res_1 if conf_1 >= conf_2 else res_2
        print(res_1, res_2)
        time.sleep(0.5)
        return final_result
    else:
        return ''

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
        raw_df['classify_result'] = raw_df.apply(f1,
                                                 select_cols=sel_cols,
                                                 expr=expr,
                                                 model=sym_spell,
                                                 axis=1)

        # raw_df = raw_df.drop(columns=['classify_result'])
        raw_df.to_csv(f"{fn}_output{ext}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, help="JSON file", default="config.json")
    args, _ = parser.parse_known_args()
    args = vars(args)

    main()
