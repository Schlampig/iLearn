import re
import json
from elasticsearch import Elasticsearch
from es_re_pre import EsRE_Pre
import ipdb

INDEX_NAME = "tempor_es_base"
FIELD_NAME = "my_doc"
es = Elasticsearch([{"host":"localhost", "port":9207}])
if es.indices.exists(index=INDEX_NAME):
    es.indices.delete(index=INDEX_NAME)
es.indices.create(index=INDEX_NAME, ignore=400)


def es_build(lst_text):
    # insert temporary lst_text into es database
    # input: lst_text = [piece_1, piece_2, ...], where piece_i is a long string, e.g., sentence or paragraph(type: text for es)
    # output: Boolean (True for sucessfully update es, otherwise False)
    global INDEX_NAME, FIELD_NAME, es
    es.delete_by_query(index=INDEX_NAME, body={"query": {"bool": {"should": {"match_all": {}}}}}, ignore=400)
    es.indices.create(index=INDEX_NAME, ignore=400)
    es.indices.refresh(index=INDEX_NAME)
    
    mapping = {
        "properties":{
            FIELD_NAME:{
                "type": "text",
                "analyzer": "ik_smart",
                "search_analyzer": "ik_smart"
            }
        }
    }
    es.indices.put_mapping(index=INDEX_NAME, body=mapping)

    for text_now in lst_text:
        text_now = {FIELD_NAME: text_now}
        es.index(index=INDEX_NAME, body=text_now)
    if es.indices.exists(index=INDEX_NAME):
        return True
    else:
        return False


def es_search(lst_query, candi_size=20):
    # search eligible pieces according to the given requirements (lst_query)
    # input: lst_query = [query_1, query_2, ...], where query is a string, e.g., question, description
    #      candi_size is an int for number of returned results
    # output: result is an es-based json-like data structure
    global INDEX_NAME, FIELD_NAME, es
    # preprocess matching rules
    lst_should = list(map(lambda x: {"match": {FIELD_NAME: {"query": x}}}, lst_query))
    if len(lst_should) == 0:
        lst_should = {"match_all": {}}
    # define searching rules
    rule = {
        "query": {
            "bool": {
                "should": lst_should
            }
        }
    }
    # search for candidates
    result = es.search(index=INDEX_NAME, size=candi_size, body=rule)
    return result


class EsRE(object):
    
    def __init__(self):
        self.candi_size = 5
        
    def run(self, lst_q, doc):
        global FIELD_NAME
        prepro = EsRE_Pre()
        lst_text = prepro.run(doc)
        lst_candidate = list()
        # build es and search based on the built es
        if es_build(lst_text):
            res_now = es_search(lst_q, candi_size=self.candi_size)
            lst_candidate = [(x.get("_source").get(FIELD_NAME), x.get("_score")) for x in res_now.get("hits", {}).get("hits", [])]
        return lst_candidate
    

if __name__ == "__main__":
    
    c = """1998年，拉里·佩奇和谢尔盖·布林在美国斯坦福大学的学生宿舍内共同开发了谷歌在线搜索引擎，并迅速传播给全球的信息搜索者；8月7日，谷歌公司在美国加利福尼亚州山景城以私有股份公司的型式创立。同年，发明Google PageRank专利。2001年9月，谷歌的网页评级机制PageRank被授予了美国专利。专利正式地被颁发给斯坦福大学，拉里·佩奇作为发明人列于其中。2004年2月，因为雅虎放弃使用谷歌搜索引擎而决定独立开发自己的搜索引擎，谷歌的市场份额较前跌落。2004年8月19日，谷歌公司在纳斯达克上市，成为公有股份公司。2005年7月19日，谷歌宣布在中国设立研发中心。2005年12月20日，谷歌宣布斥资10亿美元收购互联网服务供应商“美国在线”5%的股权。2006年2月15日，谷歌在台湾地区登记成立分公司“美商科高国际有限公司”，并作为台湾服务器的域名运营商。"""
    q = ["收购摩托罗拉移动", "摩托罗拉"]
    
    es_re = EsRE()
    answers = es_re.run(q, c)
    for ans in answers:
        print(ans)
        print()
        
