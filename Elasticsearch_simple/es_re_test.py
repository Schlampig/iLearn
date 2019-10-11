import os
from pprint import pprint
from docx import Document
from es_re import *


load_path = "../your_test_file_path"  # the file should be in .docx format
doc = Document(load_path)
lst_doc = [para.text for para in doc.paragraphs if len(para.text) > 0]
doc = "\n".join(lst_doc)
print("="*30)
print("Doc Size: ", len(doc))

lst_q = ["原始权益人", "资产服务机构"]
es_re = EsRE()
answers = es_re.run(lst_q, doc)

print("="*30)
for ans in answers:
    print(ans)
    print()
print("="*30)
    
    
