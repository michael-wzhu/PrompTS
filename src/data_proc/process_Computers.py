import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
下面的一组时间序列是设备采集到的某个家庭的电脑用电行为数据：<ts-data-blank-sep>，其中数据点采用空格间隔, 包含720个数据点，为传感器以2分钟一次的频率连续获取24小时采集得到。
请推断所采集的电脑设备类型。
<options>
可能的电脑类型包括：台式机、笔记本。
请根据上述类型选择回答，不要回答其他内容
<response>
"""

list_templates_by_chatgpt = [
    """<query>
The following set of time series represents the electricity consumption data of a computer in a household: <ts-data-blank-sep>. The data points are separated by spaces and consist of 720 data points collected continuously over 24 hours at a frequency of one data point every 2 minutes.
Please infer the type of computer device that was being monitored.
<options>
Possible computer types include: Desktop, Laptop.
Please choose your answer based on the provided options and do not include any other content.
<response>
""",
"""<query>
The provided dataset includes a sequence of time series that represents the electricity consumption of a computer in a household: <ts-data-blank-sep>. The data points are separated by spaces and consist of 720 data points recorded continuously over a 24-hour period, with a frequency of one data point captured every 2 minutes.
Please make an inference regarding the type of computer device that was under monitoring.
<options>
Possible computer types include: Desktop, Laptop.
Please select your answer from the given options and refrain from including any additional content.
<response>
""",
    """<query>
下面是一组时间序列数据，记录了某个家庭电脑的用电行为：<ts-data-blank-sep>。数据点之间用空格分隔，共有720个数据点，连续记录了24小时内每2分钟的电力消耗情况。
请根据这些数据推测所监测的电脑类型。
<options>
可能的电脑类型包括：台式机、笔记本。
请从提供的选项中选择，并不要提供其他内容。
<response>
""",
    """<query>
以下是一组时间序列数据，记录了某个家庭电脑在24小时内的用电情况：<ts-data-blank-sep>。数据点之间以空格分隔，共有720个数据点，每隔2分钟记录一次。这些数据反映了电脑的用电行为。
请根据这些数据推测所监测的电脑类型。
<options>
可能的电脑类型包括：台式机、笔记本。
请根据给定选项选择答案，并避免提供其他内容。
<response>
""", 


]


label_map = {
    "1": "Desktop",
    "2": "Laptop"
}


label_map_zh = {
    "1": "台式机",
    "2": "笔记本",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "Computers"
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
            # assert len(ts_data_) == 96
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
                json.dumps(samp, ensure_ascii=False,) + "\n"
            )
