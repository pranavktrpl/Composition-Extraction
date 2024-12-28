"""
---------------------------------------------------------------------------------------------------------
The table extraction code has been adapted from the following repository: https://github.com/olivettigroup/table_extractor
The license for the same is provided below.
---------------------------------------------------------------------------------------------------------
MIT License

Copyright (c) 2018

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
---------------------------------------------------------------------------------------------------------
"""

import traceback
from bs4 import BeautifulSoup
import unidecode
import sys

from scipy import stats
# from html.parser import HTMLParser #Depreciated for Py-3
import html  # Import the html module for unescaping
import traceback


import traceback
from bs4 import BeautifulSoup
import unidecode
import sys

from scipy import stats
import html  # Import the html module for unescaping

class TableExtractor(object):
    def __init__(self):
        print("Initializing TableExtractor...")
        return

    def get_caption(self, table, format):
        print("Attempting to extract caption...")
        if format == 'xml':
            if '10.1016' in self.doi:
                caption = table.find('caption')
                if caption:
                    print(f"Caption found: {caption.text.strip()[:50]}...")
                caption, ref = self._search_for_reference(caption, format)
                caption = unidecode.unidecode(html.unescape(caption.text)).strip()
                return caption, ref
            elif '10.1021' in self.doi:
                caption = table.find('title')
                if not caption:
                    print("Title not found, checking parent...")
                    up = table.parent
                    caption = up.find('caption') or table.find('title')
                if caption:
                    print(f"Caption found: {caption.text.strip()[:50]}...")
                caption, ref = self._search_for_reference(caption, format)
                caption = unidecode.unidecode(html.unescape(caption.text)).strip()
                return caption, ref
        else:
            raise NotImplementedError
        print("No caption found.")
        return '', []

    def get_footer(self, table, format):
        print("Attempting to extract footer...")
        footer_dict = dict()
        if format == 'xml':
            if '10.1016' in self.doi:
                footer = table.find_all('table-footnote')
                if footer:
                    print(f"Found {len(footer)} table footnotes.")
                    for f in footer:
                        sup = f.find('label')
                        dt = sup.text if sup else 'NA'
                        if sup:
                            f.label.decompose()
                        footer_dict[dt.strip()] = unidecode.unidecode(html.unescape(f.text)).strip()
                else:
                    footer = table.find('legend')
                    if footer:
                        print("Found legend footer.")
                        all_paras = footer.find_all('simple-para')
                        for a in all_paras:
                            sup = a.find('sup')
                            dt = sup.text if sup else 'NA'
                            if sup:
                                a.sup.decompose()
                            footer_dict[dt.strip()] = unidecode.unidecode(html.unescape(a.text)).strip()
                    else:
                        print("No footer found.")
            elif '10.1021' in self.doi:
                up = table.parent
                footer = up.find('table-wrap-foot')
                if footer:
                    print("Footer found under table-wrap-foot.")
                    dts, dds = footer.find_all('label'), footer.find_all('p')
                    if len(dts) != len(dds):
                        print("Mismatch in footer labels and descriptions.")
                    for d, t in zip(dds, dts):
                        footer_dict[t.text.strip()] = unidecode.unidecode(html.unescape(d.text)).strip().replace('\n', ' ')
        else:
            raise NotImplementedError
        print(f"Footer extraction complete. Keys: {list(footer_dict.keys())}")
        return footer_dict

    def get_xml_tables(self, xml):
        print(f"Opening XML file: {xml}")
        all_tables = []
        all_captions = []
        all_footers = []

        try:
            with open(xml, 'r+', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'xml')
        except Exception as e:
            print(f"Failed to parse XML file: {e}")
            return all_tables, all_captions, all_footers

        tables = soup.find_all('table')
        print(f"Found {len(tables)} tables.")
        
        for idx, table in enumerate(tables):
            print(f"Processing table {idx + 1}...")
            try:
                caption = self.get_caption(table, format='xml')[0]
                footer = self.get_footer(table, format='xml')

                print(f"Caption: {caption[:50]}...") if caption else print("No caption.")
                print(f"Footer: {footer.keys()}") if footer else print("No footer.")

                tab = []
                for t in range(400):
                    tab.append([None] * 400)

                rows = table.find_all('row')
                print(f"Found {len(rows)} rows in table {idx + 1}.")
                for i, row in enumerate(rows):
                    counter = 0
                    for ent in row.find_all():
                        curr_col, beg, end, more_row = 0, 0, 0, 0
                        if ent.has_attr('colname'):
                            try:
                                curr_col = int(ent['colname'])
                            except:
                                assert ent['colname'].startswith('col')
                                curr_col = int(ent['colname'][3:])
                        if ent.has_attr('namest'):
                            try:
                                beg = int(ent['namest'])
                            except:
                                assert ent['namest'].startswith('col')
                                beg = int(ent['namest'][3:])
                        if ent.has_attr('nameend'):
                            try:
                                end = int(ent['nameend'])
                            except:
                                assert ent['nameend'].startswith('col')
                                end = int(ent['nameend'][3:])
                        if ent.has_attr('morerows'):
                            try:
                                more_row = int(ent['morerows'])
                            except:
                                assert ent['morerows'].startswith('col')
                                more_row = int(ent['morerows'][3:])
                        ent_text = unidecode.unidecode(html.unescape(ent.get_text())).strip().replace('\n', ' ')
                        if beg != 0 and end != 0 and more_row != 0:
                            print(f"Populating merged cell spanning rows and columns: {ent_text}")
                            for j in range(beg, end + 1):
                                for k in range(more_row + 1):
                                    tab[i + k][j - 1] = ent_text
                        elif beg != 0 and end != 0:
                            print(f"Populating merged cell spanning columns: {ent_text}")
                            for j in range(beg, end + 1):
                                tab[i][j - 1] = ent_text
                        elif more_row != 0:
                            print(f"Populating merged cell spanning rows: {ent_text}")
                            for j in range(more_row + 1):
                                tab[i + j][counter] = ent_text
                        elif curr_col != 0:
                            print(f"Populating specific column {curr_col}: {ent_text}")
                            tab[i][curr_col - 1] = ent_text
                        else:
                            while counter < len(tab[i]) and tab[i][counter] is not None:
                                counter += 1
                            print(f"Populating next available cell: {ent_text}")
                            tab[i][counter] = ent_text
                            counter += 1

                tab = [row for row in tab if any(row)]  # Remove empty rows
                print(f"Removed empty rows. Remaining rows: {len(tab)}")
                max_cols = max(len(row) for row in tab)
                tab = [row + [''] * (max_cols - len(row)) for row in tab]  # Standardize row lengths
                print(f"Standardized row lengths to {max_cols} columns.")
                all_tables.append(tab)
                all_captions.append(caption)
                all_footers.append(footer)

            except Exception as e:
                print(f"Failed to process table {idx + 1}: {e}")
                traceback.print_exc()

        print(f"Extraction complete. Processed {len(all_tables)} tables.")
        # print(all_tables, all_captions, all_footers)
        return all_tables, all_captions, all_footers

    def _search_for_reference(self, soup, format):
        print("Searching for references in text...")
        if format == 'xml':
            ref = soup.find_all('xref')
            tags = []
            if len(ref) == 0:
                if soup.name == 'caption':
                    print("No references found in caption.")
                    return soup, tags
                ref = soup.find_all('sup')
                for r in ref:
                    text = r.text.split(',')
                    for t in text:
                        if len(t) == 1 and t.isalpha():
                            tags.append(t)
                            soup.sup.decompose()
                print(f"References found: {tags}")
                return soup, tags
            else: 
                for r in ref:
                    if len(r.text) < 4:
                        tag = soup.xref.extract()
                        tags.append(tag.text)
                print(f"References extracted: {tags}")
                return soup, tags
        else:
            raise NotImplementedError
