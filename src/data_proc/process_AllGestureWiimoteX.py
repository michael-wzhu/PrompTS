import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
现在采用内置三轴加速度计的任天堂Wiimote遥控器来收集手势数据，下面的一组时间序列是设备采集到的一个人在做某种手势时的x 轴方向的加速度：<ts-data-blank-sep>，其中数据点采用空格间隔. "NaN"字符在这里是填充符号，没有具体含义
请问这个人在做什么手势？
<options>
可能的手势类型包括：捡东西，摇晃，向右移动，向左移动，向上移动，向下移动，向左画圈，向右画圈，向屏幕移动，远离屏幕。
请根据上述类型选择回答，不要回答其他内容
<response>
"""

list_templates_by_chatgpt = [
    """<query>
我们现在使用内置三轴加速度计的任天堂Wiimote遥控器来记录手势数据。以下时间序列是该设备在某人进行手势时在x轴方向上记录的加速度数据：<ts-data-blank-sep>，其中数据点以空格分隔。"NaN"字符在这里是填充符号，没有具体含义
请推断这个人正在做什么手势？
<options>
可能的手势类型包括：捡东西，摇晃，向右移动，向左移动，向上移动，向下移动，向左画圈，向右画圈，向屏幕移动，远离屏幕。
请从上述选项中选择一个回答！
<response>
""",
    """<query>
以任天堂Wiimote遥控器为例，我们正在使用内置的三轴加速度计来采集手势数据。以下是一个人进行某种手势时，设备在x轴方向记录的一系列时间序列：<ts-data-blank-sep>。数据点之间以空格分隔。"NaN"字符在这里是填充符号，没有具体含义
你能猜出这个人正在做什么手势吗？
<options>
可能的手势类型包括：捡东西，摇晃，向右移动，向左移动，向上移动，向下移动，向左画圈，向右画圈，向屏幕移动，远离屏幕。
请根据上述选项选择一个回答。
<response>
""",
    """<query>
我们正在采用任天堂Wiimote遥控器，其中内置了三轴加速度计，用于收集手势数据。以下时间序列代表某人在进行特定手势时在x轴方向上记录的加速度数据：<ts-data-blank-sep>。数据点之间使用空格分隔。"NaN"字符在这里是填充符号，没有具体含义
请问这个人在进行哪种手势？
<options>
可能的手势类型包括：捡东西，摇晃，向右移动，向左移动，向上移动，向下移动，向左画圈，向右画圈，向屏幕移动，远离屏幕。
请从上述类型中选择一个答案
<response>
""",
    """<query>
We are currently using a Nintendo Wiimote controller with a built-in triaxial accelerometer to collect gesture data. The following time series represents the recorded acceleration in the x-axis when a person performs a certain gesture: <ts-data-blank-sep>.
Can you determine what gesture the person is performing?
<options>
Possible gesture types include: picking up, shaking, moving right, moving left, moving up, moving down, drawing a circle to the left, drawing a circle to the right, moving towards the screen, moving away from the screen.

Please select a response from the above options:
<response>
""",
    """<query>
Let's consider the Nintendo Wiimote controller as an example. We are using its built-in triaxial accelerometer to record gesture data. The following is a time series of the recorded acceleration in the x-axis when a person performs a gesture: <ts-data-blank-sep>. The data points are separated by spaces.
Can you guess what gesture the person is performing?
<options>
Possible gesture types include: picking up, shaking, moving right, moving left, moving up, moving down, drawing a circle to the left, drawing a circle to the right, moving towards the screen, moving away from the screen.

Please choose a response from the options above:
<response>
""",
    """<query>
We are collecting gesture data using a Nintendo Wiimote controller equipped with a built-in triaxial accelerometer. The following time series represents the recorded acceleration in the x-axis when a person is performing a gesture: <ts-data-blank-sep>. 
Which gesture is the person performing?
<options>
Possible gesture types include: picking up, shaking, moving right, moving left, moving up, moving down, drawing a circle to the left, drawing a circle to the right, moving towards the screen, moving away from the screen.

Please select an answer from the above types:
<response>
""",
    """<query>
Regarding gesture data collection, we are utilizing the built-in triaxial accelerometer in the Nintendo Wiimote controller to record data. The following time series represents the recorded acceleration in the x-axis when a person performs a particular gesture: <ts-data-blank-sep>. The data points are separated by spaces.
What gesture do you think the person is performing?
<options>
Possible gesture types include: picking up, shaking, moving right, moving left, moving up, moving down, drawing a circle to the left, drawing a circle to the right, moving towards the screen, moving away from the screen.

Please determine an answer from the above types:
<response>
""",
    """<query>
We are currently using a Nintendo Wiimote controller with a built-in triaxial accelerometer to collect gesture data. The following time series represents the recorded acceleration in the x-axis when a person performs a specific gesture: <ts-data-blank-sep>. The data points are separated by spaces.
Can you identify the gesture being performed?
<options>
Possible gesture types include: picking up, shaking, moving right, moving left, moving up, moving down, drawing a circle to the left, drawing a circle to the right, moving towards the screen, moving away from the screen.

Please choose a response based on the above options:
<response>
""",

]


label_map = {
    "1": "picking up",
    "2": "shaking",
    "3": "moving right",
    "4": "moving left",
    "5": "moving up",
    "6": "moving down",
    "7": "drawing a circle to the left",
    "8": "drawing a circle to the right",
    "9": "moving towards the screen",
    "10": "moving away from the screen",
}

"""
捡东西，摇晃，向右移动，向左移动，向上移动，向下移动，向左画圈，向右画圈，向屏幕移动，远离屏幕
"""
label_map_zh = {
    "1": "捡东西",
    "2": "摇晃",
    "3": "向右移动",
    "4": "向左移动",
    "5": "向上移动",
    "6": "向下移动",
    "7": "向左画圈",
    "8": "向右画圈",
    "9": "向屏幕移动",
    "10": "远离屏幕",
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "AllGestureWiimoteX"
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
                json.dumps(samp, ensure_ascii=False) + "\n"
            )
