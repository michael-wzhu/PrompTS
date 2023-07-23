import copy
import json
import os

task_type_prompt = "Task type: time series classfication"


template_1 = """<query>
The following time series is the outlines of hand derived from images : <ts-data-blank-sep>\n Since the data is extracted automatically by a certain algorithm, could you determine whether the output of the image outlinings is correct or incorrect?
<options>
candidate labels: correct, incorrect
<response>
"""

list_templates_by_chatgpt = [
    """<query>
We have a time series representing the outlines of hands derived from images: <ts-data-blank-sep>\n The data extraction process is automated, using a specific algorithm. I need your assistance in determining whether the output of the image outlinings is correct or incorrect.
<options>
Please provide one of the following candidate labels: "correct" or "incorrect."
<response>
    """,
    """<query>
We possess a time series that illustrates hand outlines obtained from images: <ts-data-blank-sep>\n As the data is automatically extracted using a particular algorithm, can you ascertain whether the image outlinings are accurate or not?
<options>
You can choose from the candidate labels: "correct" or "incorrect."
<response>
    """,
    """<query>
Presented here is a time series displaying hand outlines obtained from images: <ts-data-blank-sep>\n Since an algorithm automatically extracted this data, could you help determine if the image outlinings are right or wrong?
<options>
Please select one of the candidate labels: "correct" or "incorrect."
<response>
    """,
    """<query>
We have a time series that outlines hands obtained from images: <ts-data-blank-sep>\n Given that the data is extracted using an automated algorithm, can you judge whether the image outlinings are accurate or not?
<options>
Choose one from the candidate labels: "correct" or "incorrect."
<response>
    """,
    """<query>
The following time series represents hand outlines derived from images: <ts-data-blank-sep>\n As the data extraction is automated using a specific algorithm, we need to determine if the image outlinings are correct or incorrect.
<options>
Kindly provide one of the candidate labels: "correct" or "incorrect."
<response>
    """,
    """<query>
Displayed is a time series depicting hand outlines extracted from images: <ts-data-blank-sep>\n Since the data is obtained automatically through an algorithm, can you decide whether the image outlinings are accurate or not?
<options>
You have the option to select either "correct" or "incorrect" as the candidate labels.
<response>
    """,

    
    """<query>
我们有一个时间序列，展示了从图像中得出的手部轮廓：<ts-data-blank-sep>\n由于数据是通过某种算法自动提取的，您能否确定图像轮廓的输出是正确还是错误？
<options>
候选标签：正确，错误
<response>
""",
    """<query>
这是一个展示从图像中提取的手部轮廓的时间序列：<ts-data-blank-sep>\n由于数据是通过自动算法提取的，您能否帮助确定图像轮廓的准确性？
<options>
请选择以下候选标签之一："正确" 或 "错误"。
<response>
""",
    """<query>
以下时间序列显示了从图像中获取的手部轮廓：<ts-data-blank-sep>\n由于数据是通过特定算法自动提取的，您能否判断图像轮廓的准确性？
<options>
请从以下候选标签中选择："正确" 或 "错误"。
<response>
""",
]

label_map = {
    "0": "correct",
    "1": "incorrect",
}
label_map_zh = {
    "0": "正确",
    "1": "错误",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "HandOutlines"
to_folder = os.path.join(
    "datasets/prompt_datasets", f"{task_name}"
)
os.makedirs(to_folder, exist_ok=True)

for mode in ["TRAIN", "TEST"]:
    list_datas = []
    with open(f"datasets/UCRArchive_2018/{task_name}/{task_name}_{mode}.tsv", "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            label_ = row.split("\t")[0]
            label_name_ = label_map[label_]
            label_zh_name_ = label_map_zh[label_]

            ts_data_ = row.split("\t")[1: ]
            assert len(ts_data_) == 2709
            ts_data_blank_sep = " ".join(ts_data_)
            ts_data_comma_sep = ",".join(ts_data_)

            for template_ in list_templates:
                template_ = copy.copy(template_)

                # 确认数据怎么分割的
                if "comma-sep" in template_:
                    template_.replace(
                        "<ts-data-comma-sep>",
                        ts_data_comma_sep
                    )
                else:
                    template_.replace(
                        "<ts-data-blank-sep>",
                        ts_data_blank_sep
                    )

                # 确认中文还是英文
                target = ""
                if "序列" in template_:
                    target = label_zh_name_
                else:
                    target = label_name_

                list_datas.append(
                    {
                        "original_data": (label_name_, ts_data_blank_sep),
                        "target": target,
                        "input": template_,
                        "task_name": task_name,
                        "task_type": "classification",
                    }
                )


    with open(os.path.join(to_folder, f"{mode}.json"), "w", encoding="utf-8") as f:
        for samp in list_datas:
            f.write(
                json.dumps(samp, ensure_ascii=False) + "\n"
            )
