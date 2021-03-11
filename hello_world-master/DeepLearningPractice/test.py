import re

input = input()

pattern = r"\d + |S|D|T|\*|\#"
dict_bonus_option = {"S" : "**1", 'D': "**2", "T": "**3", "#": '*-1', '*': '*2'}

array_input = re.findall(pattern, input)
array_result = list()
