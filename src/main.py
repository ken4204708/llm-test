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
import pdb


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
    for category in response.categories:
        # Get the name of the category representing the document.
        # See the predefined taxonomy of categories:
        # https://cloud.google.com/natural-language/docs/categories
        print(f"Category name: {category.name}")
        # Get the confidence. Number representing how certain the classifier
        # is that this category represents the provided text.
        print(f"Confidence: {category.confidence}")
    if response:
        confidences = [x.confidence for x in response.categories]
        category_with_highest_confidence = response.categories[np.argmax(confidences)].name
    else:
        category_with_highest_confidence = ''
    return category_with_highest_confidence


# [END language_classify_text]

def read_csv_file(file_name, dropna_rule = []):
    raw_df = pd.read_csv(file_name)
    print(f'''There are {len(raw_df)} rows in file {file_name}''')
    if len(dropna_rule) == 0:
        print(f'''Removing nan fields in the {dropna_rule}''')
        raw_df = raw_df.dropna(subset = dropna_rule)
        print(f'''There are {len(raw_df)} rows in file {file_name} after drop nan fields''')
    return raw_df

def f1(df):
    return sample_classify_text(df['Name'])

def main():
    file_names = ['Product.csv', 'Campaign.csv']
    dropna_rules = [['Name'], []]
    file_name = file_names[0]
    dropna_rule = dropna_rules[0]
    raw_df = read_csv_file(file_name, dropna_rule)
    raw_df['classify_result'] = raw_df['Name'].apply(f1)
    

# def main():
#     import argparse

#     parser = argparse.ArgumentParser()
#     parser.add_argument(
#         "--text_content",
#         type=str,
#         default="That actor on TV makes movies in Hollywood and also stars in a variety of popular new TV shows.",
#     )
#     args = parser.parse_args()

#     sample_classify_text(args.text_content)


if __name__ == "__main__":
    main()
