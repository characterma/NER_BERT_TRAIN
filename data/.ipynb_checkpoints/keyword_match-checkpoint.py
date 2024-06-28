#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author:Shelly
# @time:2023/4/21:10:53
#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Quincy Liang"
__email__ = "lguanq@gmail.com"

import os

from flashtext import KeywordProcessor



class FixedKeywordProcessor(KeywordProcessor):
    def __init__(self, case_sensitive=False):
        super().__init__(case_sensitive)

    def extract_keywords(self, sentence, span_info=False):
        """Searches in the string for all keywords present in corpus.
        Keywords present are added to a list `keywords_extracted` and returned.

        Args:
            sentence (str): Line of text where we will search for keywords

        Returns:
            keywords_extracted (list(str)): List of terms/keywords found in sentence that match our corpus

        Examples:
            # >>> from flashtext import KeywordProcessor
            # >>> keyword_processor = KeywordProcessor()
            # >>> keyword_processor.add_keyword('Big Apple', 'New York')
            # >>> keyword_processor.add_keyword('Bay Area')
            # >>> keywords_found = keyword_processor.extract_keywords('I love Big Apple and Bay Area.')
            # >>> keywords_found
            # >>> ['New York', 'Bay Area']

        """
        keywords_extracted = []
        if not sentence:
            # if sentence is empty or none just return empty list
            return keywords_extracted
        if not self.case_sensitive:
            sentence = sentence.lower()
        current_dict = self.keyword_trie_dict
        sequence_start_pos = 0
        sequence_end_pos = 0
        reset_current_dict = False
        idx = 0
        sentence_len = len(sentence)
        while idx < sentence_len:
            char = sentence[idx]
            # when we reach a character that might denote word end
            if char not in self.non_word_boundaries:

                # if end is present in current_dict
                if self._keyword in current_dict or char in current_dict:
                    # update longest sequence found
                    sequence_found = None
                    longest_sequence_found = None
                    is_longer_seq_found = False
                    if self._keyword in current_dict:
                        sequence_found = current_dict[self._keyword]
                        longest_sequence_found = current_dict[self._keyword]
                        sequence_end_pos = idx

                    # re look for longest_sequence from this position
                    if char in current_dict:
                        current_dict_continued = current_dict[char]

                        idy = idx + 1
                        while idy < sentence_len:
                            inner_char = sentence[idy]
                            if inner_char not in self.non_word_boundaries and self._keyword in current_dict_continued:
                                # update longest sequence found
                                longest_sequence_found = current_dict_continued[self._keyword]
                                sequence_end_pos = idy
                                is_longer_seq_found = True
                            if inner_char in current_dict_continued:
                                current_dict_continued = current_dict_continued[inner_char]
                            else:
                                break
                            idy += 1
                        else:
                            # end of sentence reached.
                            if self._keyword in current_dict_continued:
                                # update longest sequence found
                                longest_sequence_found = current_dict_continued[self._keyword]
                                sequence_end_pos = idy
                                is_longer_seq_found = True
                        if is_longer_seq_found:
                            idx = sequence_end_pos - 1  # fix the bug
                    current_dict = self.keyword_trie_dict
                    if longest_sequence_found:
                        keywords_extracted.append((longest_sequence_found, sequence_start_pos, idx))
                    reset_current_dict = True
                else:
                    # we reset current_dict
                    current_dict = self.keyword_trie_dict
                    reset_current_dict = True
            elif char in current_dict:
                # we can continue from this char
                current_dict = current_dict[char]
            else:
                # we reset current_dict
                current_dict = self.keyword_trie_dict
                reset_current_dict = True
                # skip to end of word
                idy = idx + 1
                while idy < sentence_len:
                    char = sentence[idy]
                    if char not in self.non_word_boundaries:
                        break
                    idy += 1
                idx = idy
            # if we are end of sentence and have a sequence discovered
            if idx + 1 >= sentence_len:
                if self._keyword in current_dict:
                    sequence_found = current_dict[self._keyword]
                    keywords_extracted.append((sequence_found, sequence_start_pos, sentence_len))
            idx += 1
            if reset_current_dict:
                reset_current_dict = False
                sequence_start_pos = idx
        if span_info:
            keywords_extracted = [(value[0], value[1], value[2] + 1) for value in keywords_extracted]
            return keywords_extracted
        return [value[0] for value in keywords_extracted]


class KeyWordExtractor(object):
    def __init__(self, dictionary_file=None, dictionary=None, use_fixed=True, case_sensitive=True):
        if dictionary_file is not None:
            self.dictionary = [data.rstrip() for data in open(dictionary_file, encoding='utf-8')]
        elif isinstance(dictionary, list):
            self.dictionary = dictionary
        else:
            raise ValueError('the input value has problems!')
        if use_fixed:
            self.keyword_processor = FixedKeywordProcessor(case_sensitive=case_sensitive)
        else:
            self.keyword_processor = KeywordProcessor(case_sensitive=case_sensitive)
        self.add_dictionary(self.dictionary)

    def add_dictionary(self, dictionary):
        for keyword in dictionary:
            self.keyword_processor.add_keyword(keyword)

    def extract(self, text):
        result = self.keyword_processor.extract_keywords(text, span_info=True)
        return result


def get_keyword_extractor(dictionary_file=None, dictionary=None, use_fixed=True, case_sensitive=True):
    key_word_extrator = KeyWordExtractor(dictionary_file=dictionary_file, dictionary=dictionary, use_fixed=use_fixed, case_sensitive=case_sensitive)

    def extract_keyword(text):
        return key_word_extrator.extract(text)

    return extract_keyword


if __name__ == '__main__':
    # file_path = os.path.join(gPROJECT_PATH, 'conf', 'company_list.txt')
    # Extractor = get_keyword_extractor(dictionary_file=file_path)
    

    import json
    companys = list(json.load(open("/workspace/ner-starbucks/configs/company_entity.json", 'r', encoding='utf-8')).keys())
    print(f"companys: {len(companys)}， {'老乡鸡' in companys}")
    Extractor = get_keyword_extractor(dictionary=companys, use_fixed=True)
    print(Extractor('卡巴斯基优秀人员买了一个带有Wi Fi功能的索尼相机.'))
    
    text = '你喝了健力寶奶茶，吃了老乡鸡，接到通知后, Check point决定派出两名安全技术专家前往福州驻场。\n\n博客地址:\n\nhttps://www.fireeye.com/blog/threat-research' \
           '.html\n\n賽門鐵克技术团队\n\n(\n\nhttp://blogs.360.cn/\n\n國外戰場\n\n卡巴斯基\n\n(俄羅斯)\n\n全球研究與分析團隊 (GReAT) 成立于 2008 ' \
           '年，是卡巴斯基實驗室的核心運營團隊，該團隊發現了全球的 APT、網絡間諜活動、主要惡意軟件、勒索軟件和地下網絡罪犯發展趨勢。'

    print(Extractor(text))
    Extractor = get_keyword_extractor(dictionary=['优秀', '人员', 'Checkpoint'], use_fixed=True)
    print(Extractor('Checkpoint优秀人员买了一个带有Wi Fi功能的iphone 6.'))
