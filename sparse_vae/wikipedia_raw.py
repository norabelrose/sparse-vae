# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Wikipedia dataset containing cleaned articles of all languages."""

import bz2
import codecs
import json
import xml.etree.cElementTree as etree
from urllib.request import urlopen

import datasets

logger = datasets.logging.get_logger(__name__)

_CITATION = """\
@ONLINE {wikidump,
    author = {Wikimedia Foundation},
    title  = {Wikimedia Downloads},
    url    = {https://dumps.wikimedia.org}
}
"""

_DESCRIPTION = """\
Wikipedia dataset containing cleaned articles of all languages.
The datasets are built from the Wikipedia dump
(https://dumps.wikimedia.org/) with one split per language. Each example
contains the content of one full Wikipedia article with cleaning to strip
markdown and unwanted sections (references, etc.).
"""

_LICENSE = (
    "This work is licensed under the Creative Commons Attribution-ShareAlike "
    "3.0 Unported License. To view a copy of this license, visit "
    "http://creativecommons.org/licenses/by-sa/3.0/ or send a letter to "
    "Creative Commons, PO Box 1866, Mountain View, CA 94042, USA."
)

_BASE_URL_TMPL = "https://dumps.wikimedia.org/{lang}wiki/{date}/"
_INFO_FILE = "dumpstatus.json"


class WikipediaConfig(datasets.BuilderConfig):
    """BuilderConfig for Wikipedia."""

    def __init__(self, language=None, date=None, **kwargs):
        """BuilderConfig for Wikipedia.
        Args:
          language: string, the language code for the Wikipedia dump to use.
          date: string, date of the Wikipedia dump in YYYYMMDD format. A list of
            available dates can be found at https://dumps.wikimedia.org/enwiki/.
          **kwargs: keyword arguments forwarded to super.
        """
        super(WikipediaConfig, self).__init__(
            name="{0}.{1}".format(date, language),
            description=f"Wikipedia dataset for {language}, parsed from {date} dump.",
            **kwargs
        )
        self.date = date
        self.language = language


_VERSION = datasets.Version("1.0.0", "")


class WikipediaRaw(datasets.GeneratorBasedBuilder):
    """Wikipedia dataset."""

    # Use mirror (your.org) to avoid download caps.
    BUILDER_CONFIG_CLASS = WikipediaConfig
    BUILDER_CONFIGS = [
        WikipediaConfig(
            version=_VERSION,
            language='en',
            date="20200501",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features({"title": datasets.Value("string"), "text": datasets.Value("string")}),
            # No default supervised_keys.
            supervised_keys=None,
            homepage="https://dumps.wikimedia.org",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager, pipeline):
        def _base_url(lang):
            return _BASE_URL_TMPL.format(lang=lang.replace("-", "_"), date=self.config.date)

        lang = self.config.language

        info_url = _base_url(lang) + _INFO_FILE
        # Use dictionary since testing mock always returns the same result.
        downloaded_files = dl_manager.download_and_extract({"info": info_url})

        xml_urls = []
        # total_bytes = 0
        with open(downloaded_files["info"], encoding="utf-8") as f:
            dump_info = json.load(f)
        multistream_dump_info = dump_info["jobs"]["articlesmultistreamdump"]
        assert (
                multistream_dump_info["status"] == "done"
        ), "Specified dump (%s) multistream status is not 'done': %s" % (
            _base_url(lang),
            multistream_dump_info["status"],
        )

        for fname, info in multistream_dump_info["files"].items():
            if ".xml" not in fname:
                continue
            # total_bytes += info["size"]
            xml_urls.append(_base_url(lang) + fname)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={"urls": xml_urls}
            )
        ]

    def _generate_examples(self, urls):
        for url in urls:
            yield from self._generate_from_url(url)

    @staticmethod
    def _generate_from_url(url):
        # Decompress the bz2 file as it is streamed in
        with urlopen(url) as response:
            f = bz2.BZ2File(response)

        utf_f = codecs.getreader("utf-8")(f)
        context = etree.iterparse(utf_f, events=("end",))
        for unused_event, elem in context:
            if not elem.tag.endswith("page"):
                continue

            namespace = elem.tag[:-4]
            title = elem.find("./{0}title".format(namespace)).text
            ns = elem.find("./{0}ns".format(namespace)).text
            id_ = elem.find("./{0}id".format(namespace)).text

            # Filter pages that are not in the "main" namespace.
            if ns != "0":
                elem.clear()
                continue

            raw_content = elem.find("./{0}revision/{0}text".format(namespace)).text
            elem.clear()

            # Filter redirects.
            if raw_content is None or raw_content.lower().startswith("#redirect"):
                continue

            yield id_, title, raw_content
