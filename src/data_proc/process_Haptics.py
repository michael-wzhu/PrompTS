import copy
import json
import os

task_type_prompt = "Task type: time series classfication"


template_1 = """<query>
下面的时间序列来自一个人用手在触摸屏上输入密码的X轴移动位置：<ts-data-comma-sep>。数据点之间用空格分隔。
你能帮忙判断一下这个时间序列代表的是哪个人吗？
<options>
可能的人员包括：人员1，人员2，人员3，人员4，人员5。
请从上述选项中选择一个答案。
<response>
"""

list_templates_by_chatgpt = [
    """<query>
请你帮忙分析一下以下时间序列，该序列是一个人在触摸屏上输入密码时X轴的移动位置：<ts-data-comma-sep>。数据点之间使用空格分隔。你能告诉我这个时间序列可能代表的是哪个人吗？
<options>
给出的选项包括人员1、人员2、人员3、人员4和人员5，请从中选择一个答案。
<response>
""",
    """<query>
Can you please help determine which person the following time series represents? The time series is based on the X-axis movement position as a person enters their password on a touchscreen using their hand. The time sequence is as follows: <ts-data-comma-sep>. The data points are separated by spaces. 
<options>
Person 1, Person 2, Person 3, Person 4, Person 5. Please select one answer from the given options.
<response>
""",
    """<query>
Could you assist in identifying which individual is represented by the provided time series? The time series corresponds to the movement of the X-axis position when a person inputs their password on a touchscreen with their hand. The time sequence is as follows: <ts-data-comma-sep>. Each data point is separated by spaces. 
<options>
The available options are Person 1, Person 2, Person 3, Person 4, and Person 5. Kindly choose an answer.
<response>
""",
    """<query>
Would you mind helping determine the person represented by the given time series? The time series depicts the movement of the X-axis position while someone enters their password on a touchscreen using their hand. The time sequence is as follows: <ts-data-comma-sep>. The data points are separated by spaces.
<options>
 You can select either Person 1, Person 2, Person 3, Person 4, or Person 5 as the correct answer.
<response>
""",
    """<query>
Can you please assist in identifying which individual is indicated by the presented time series? The time series corresponds to the X-axis movement position when a person is inputting their touchscreen password using their hand. The time sequence is as follows: <ts-data-comma-sep>. Each data point is separated by spaces. 
<options>
The available choices are Person 1, Person 2, Person 3, Person 4, and Person 5. Kindly select your answer.
<response>
""",
    """<query>
 Could you help determine which person is represented by the provided time series, please? The time series illustrates the movement of the X-axis position as an individual enters their password on a touchscreen using their hand. The time sequence is as follows: <ts-data-comma-sep>. The data points are separated by spaces. 
<options>
The available options to choose from are Person 1, Person 2, Person 3, Person 4, and Person 5. Kindly make a selection.
<response>
""",
    """<query>
Would you be able to help ascertain the identity of the person represented by the following time series? The time series portrays the X-axis movement position when a person inputs their password on a touchscreen using their hand. The time sequence is as follows: <ts-data-comma-sep>. Each data point is separated by spaces. 
<options>
Among the options provided, namely Person 1, Person 2, Person 3, Person 4, and Person 5, please choose an answer.
<response>
""",
]

label_map = {
    "1": "Person 1",
    "2": "Person 2",
    "3": "Person 3",
    "4": "Person 4",
    "5": "Person 5",
}
label_map_zh = {
    "1": "人员1",
    "2": "人员2",
    "3": "人员3",
    "4": "人员4",
    "5": "人员5",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "Haptics"
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
            assert len(ts_data_) == 1092
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
