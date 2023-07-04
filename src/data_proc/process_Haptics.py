import copy
import json
import os

task_type_prompt = "Task type: time series classfication"


template_1 = """<query>
下面的时间序列来自两个人用手在触摸屏上输入密码的X轴移动位置：<ts-data-blank-sep>。数据点之间用空格分隔，两个人的数据用"NAN"字符间隔。
你能帮忙判断一下这个时间序列代表的是同一个人吗？
<response>
"""

list_templates_by_chatgpt = [
    """<query>
给出下列时间序列的X轴移动位置，分别来自于两个人在触摸屏上输入密码时的手的移动：<ts-data-blank-sep>。数据点之间由空格分隔，两个人的数据由"NAN"字符分隔。能够判断这个时间序列是否代表同一个人吗？
<response>
""",
    """<query>
Can you help determine whether the time sequence represents the same person? The following time series is derived from the X-axis movement positions of two individuals inputting their password on a touchscreen: <ts-data-blank-sep>. Data points are separated by spaces, with the_nan_character representing the separation between the two individuals. 
<response>
""",
    """<query>
Evaluate whether the X-axis movement positions of the provided time series: <ts-data-blank-sep>, which correspond to the hand movements of two individuals while inputting passwords on a touchscreen, indicate that it represents the same person.
<response>
""",
    """<query>
Would you be able to assist in determining if the time sequence represents the same person? The given time series is generated from the X-axis movement positions of two individuals during password entry on a touchscreen: <ts-data-blank-sep>. Data points are separated by spaces, with the_nan_character denoting the separation between the two individuals. 
<response>
""",
    """<query>
Give the X-axis shift positions for the following time series, which come from the hand movements of two individuals while entering passwords on a touchscreen: <ts-data-blank-sep>. The data points are separated by spaces, and the data from the two individuals are separated by the NAN character. Can you determine if this time series represents the same person?
<response>
""",
    """<query>
Analyze the X-axis shift positions for the provided time series: <ts-data-blank-sep>, which originated from two separate individuals entering passwords using touchscreen input. Determine if these data points represent the same person.
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
    # 5次一循环
    num_cy = 5
    que_row = []
    with open(f"datasets/UCRArchive_2018/{task_name}/{task_name}_{mode}.tsv", "r", encoding="utf-8") as f:
        num_cy_tem = num_cy
        num_cy_cy = 0
        for row in f:
            if num_cy_tem >= 0:
                row = row.strip()
                if not row:
                    continue
                que_row.append(row)
                num_cy_tem -= 1
            else:
                row = row.strip()
                if not row:
                    continue
                que_row[num_cy_cy] = row
                num_cy_cy += 1
                num_cy_cy = num_cy_cy % num_cy


                label_ = que_row[num_cy_cy].split("\t")[0]
                ts_data_ = que_row[num_cy_cy].split("\t")[1:]
                assert len(ts_data_) == 1092

                num_tem = num_cy_cy + 1
                while( (num_tem % num_cy) != num_cy_cy):
                    label_name_ = label_map[label_]
                    ts_data_blank_sep = " ".join(ts_data_)
                    ts_data_comma_sep = ",".join(ts_data_)
                    ts_data_blank_sep += "NAN"
                    ts_data_comma_sep += "NAN"

                    row_tem = que_row[num_tem].strip()
                    label_tem = que_row[num_tem].split("\t")[0]
                    label_name_tem = label_map[label_tem]

                    label_name_ = label_name_ + " and " + label_name_tem

                    ts_data_tem = que_row[num_tem].split("\t")[1:]
                    assert len(ts_data_tem) == 1092
                    ts_data_blank_sep += " ".join(ts_data_tem)
                    ts_data_comma_sep += ",".join(ts_data_tem)

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
                            if(label_ == label_tem):
                                target = "同一个人"
                            else:
                                target = "不同的人"
                        else:
                            if(label_ == label_tem):
                                target = "Same person"
                            else:
                                target = "Different person"

                        list_datas.append(
                            {
                                "original_data": (label_name_, ts_data_blank_sep),
                                "target": target,
                                "input": template_,
                                "task_name": task_name,
                                "task_type": "classification",
                            }
                        )
                    num_tem += 1
                    num_tem = num_tem % num_cy

    with open(os.path.join(to_folder, f"{mode}.json"), "w", encoding="utf-8") as f:
        for samp in list_datas:
            f.write(
                json.dumps(samp, ensure_ascii=False) + "\n"
            )
