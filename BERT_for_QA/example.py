import requests

url = "http://xxx.xxx.xxx/"

payload = "{\"context\": \"星空凛是日本二次元偶像企划《lovelive!》的主要人物之一。15岁。现读高中一年级。在体育会系中一向开朗活泼，与其闷闷不乐不如身体先行动起来的类型。自己对于偶像活动最初并没有什么热情，起初想加入田径部，后来在帮助小泉花阳加入μ's之后受到邀请加入了μ's。在动画第一季中，因为高坂穗乃果生病而退出LoveLive！的比赛。在第二季中与其他八人一起再次参加LoveLive！并荣获冠军。\",\"question\":\"星空凛多少岁？\"}".encode("utf-8")

response = requests.request("POST", url, data=payload)

print(response.text)
