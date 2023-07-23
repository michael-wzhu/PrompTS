import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
下面给定一组时间序列：<ts-data-blank-sep>，它是从板球表演运动员手部佩戴的三维加速度传感器中获取的Y轴方向加速度数据。其中数据点采用空格间隔，加入了低方差白噪声。
请推断运动员可能正在进行的动作。
<options>
可能的动作包括：取消发球，死球，四分，最后一小时, 触身得分，无效球，短打，出界，罚分，六分，电视回放和偏球。
请根据上述类型选择回答，不要回答其他内容
<response>
"""

list_templates_by_chatgpt = [
    """<query>
下面给定一组时间序列：<ts-data-blank-sep>，它是从板球表演运动员手部佩戴的三维加速度传感器中获取的Y轴方向加速度数据。其中数据点采用空格间隔，加入了低方差白噪声。
请推断运动员可能正在进行的动作。
<options>
可能的动作包括：取消发球，死球，四分，最后一小时, 触身得分，无效球，短打，出界，罚分，六分，电视回放和偏球。
请根据上述类型选择回答，不要回答其他内容
<response>
""",
    """<query>
The given set of time series, <ts-data-blank-sep>, represents the Y-axis acceleration data obtained from accelerometer sensors worn on the hand of a cricket performing athlete. The data points are separated by spaces and low variance white noise was appended.
Please infer the possible action the athlete might be performing.
<options>
Possible actions include: Cancel Call, Dead Ball, Four, Last Hour, Leg Bye, No Ball, One Short, Out, Penalty Runs, Six, TV Replay, and Wide.
Please select your response based on the given options and do not provide any other information.
<response>
""",
    """
<query>
The provided time series data, <ts-data-blank-sep>, corresponds to the Y-axis acceleration readings collected from accelerometer sensors placed on the wrist of a cricket player during their performance. The data points are separated by spaces and have been augmented with low variance white noise.
Your task is to determine the potential action being performed by the player.
<options>
Possible actions include: Cancel Call, Dead Ball, Four, Last Hour, Leg Bye, No Ball, One Short, Out, Penalty Runs, Six, TV Replay, and Wide.
Please select your response from the provided options and refrain from providing any additional information.
<response>
""",


]


label_map_zh = {
    "1": "取消发球",
    "2": "死球",
    "3": "四分",
    "4": "最后一小时",
    "5": "触身得分",
    "6": "无效球",
    "7": "短打",
    "8": "出界",
    "9": "罚分",
    "10": "六分",
    "11": "电视回放",
    "12": "偏球"
}

label_map = {
    "1": "Cancel Call",
    "2": "Dead Ball",
    "3": "Four",
    "4": "Last Hour",
    "5": "Leg Bye",
    "6": "No Ball",
    "7": "One Short",
    "8": "Out",
    "9": "Penalty Runs",
    "10": "Six",
    "11": "TV Replay",
    "12": "Wide"
}



list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "CricketY"
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
