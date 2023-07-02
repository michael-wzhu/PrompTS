import copy
import json
import os

task_type_prompt = "Task type: time series classfication"

template_1 = """<query>
我们需要通过鱼的轮廓数据来判断鱼的种类，下面的一组时间序列是某一种鱼的轮廓数据：<ts-data-blank-sep>，其中数据点以空格分隔，请问这条鱼具体是哪一种鱼种？
<options>
鱼种可能是：奇努克鲑，冬季银鲑，棕鳟，博纳维尔切喉鳟，科罗拉多河切喉鳟，黄石公园切喉鳟，山地柱白鲑。
请根据上述鱼种选择一个回答，不回复其他内容
<response>
"""

list_templates_by_chatgpt = [
    """<query>
我们需要通过一条鱼的轮廓数据来判断这条鱼的种类，以下时间序列是某一种鱼的轮廓数据：<ts-data-blank-sep>，其中数据点以空格分隔，请判断这条鱼属于哪一种鱼种？
<options>
鱼种可能是：奇努克鲑，冬季银鲑，棕鳟，博纳维尔切喉鳟，科罗拉多河切喉鳟，黄石公园切喉鳟，山地柱白鲑。
请根据上述鱼种选择一个回答。
<response>
""",
    """<query>
我们需要通过鱼的轮廓数据来判断鱼的种类，下面的一组时间序列是某一种鱼的轮廓数据：<ts-data-blank-sep>，其中数据点以空格分隔，请问这条鱼具体是哪一种鱼种？
<options>
鱼种可能是：奇努克鲑，冬季银鲑，棕鳟，博纳维尔切喉鳟，科罗拉多河切喉鳟，黄石公园切喉鳟，山地柱白鲑。
请根据上述鱼种选择一个回答，不要回复其他内容
<response>
""",
    """<query>
我们可以利用鱼的轮廓来判断鱼的种类。下面是某一种鱼的轮廓数据的一系列时间序列：<ts-data-blank-sep>，其中数据点以空格分隔，请仔细分析时间序列数据，并选出正确的鱼种。
<options>
鱼种可能是：奇努克鲑，冬季银鲑，棕鳟，博纳维尔切喉鳟，科罗拉多河切喉鳟，黄石公园切喉鳟，山地柱白鲑。
请从上述鱼种中选择一个答案。
<response>
""",
    """<query>
We need to use the contour data to determine the type of fish, and the following time series is the contour data of a fish(separated using blank space): <ts-data-blank-sep>.
<options>
Fish species may be: Chinook salmon, winter coho, brown trout, Bonneville cutthroat, Colorado River cutthroat trout, Yellowstone cutthroat, Mountain whitefish.
Please choose one answer based on the species above and do not reply to anything else.
<response>
""",
    """<query>
We need to determine the species of a fish from its contour data. The following time series is the outline data of a fish(separated using blank space): <ts-data-blank-sep>.Please determine which species this fish belongs to? 
<options>
The fish species could be: Chinook salmon, winter coho, brown trout, Bonneville cutthroat trout, Colorado River cutthroat trout, Yellowstone cutthroat, and Mountain whitefish.
Please choose an answer based on the species above.
<response>
""",
    """<query>
We need to make a determination that identifying the species of fish by its outline data. The following set of time series is the outline data of a certain species of fish(separated using blank space): <ts-data-blank-sep>.Which specific species of fish is this fish?
<options>.
Species may be: Chinook salmon, winter coho, brown trout, Bonneville cutthroat trout, Colorado River cutthroat trout, Yellowstone cutthroat, and Mountain whitefish.
Please choose a response based on the species above and do not respond to other content.
<response>
""",
    """<query>
To analyze a set of Fish outlines used with contour matching, we possess a time series dataset(separated using blank space):<ts-data-blank-sep>.This dataset comprises the contour data of a certain species of fish.We need to determine which species this fish belongs to. Can you provide assistance in classifying the species based on this data?
<options>
Species may be: Chinook salmon, winter coho, brown trout, Bonneville cutthroat trout, Colorado River cutthroat trout, Yellowstone cutthroat, and Mountain whitefish.
Please give me the answer based on one of the options above.
<response>
""",


]

label_map = {
    "1": "Chinook salmon",
    "2": "winter coho",
    "3": "brown trout",
    "4": "Bonneville cutthroat",
    "5": "Colorado River cutthroat trout",
    "6": "Yellowstone cutthroat",
    "7": "Mountain whitefish"
}
label_map_zh = {
    "1": "奇努克鲑",
    "2": "冬季银鲑",
    "3": "棕鳟",
    "4": "博纳维尔切喉鳟",
    "5": "科罗拉多河切喉鳟",
    "6": "黄石公园切喉鳟",
    "7": "山地柱白鲑"
}


list_templates = []
list_templates.append(template_1)
list_templates.extend(list_templates_by_chatgpt)

task_name = "Fish"
to_folder = os.path.join(
    "E:\\硕\\Time_Series_Instruct-main", f"{task_name}"
)
os.makedirs(to_folder, exist_ok=True)

for mode in ["TRAIN", "TEST"]:
    list_datas = []
    with open(f"E:\\硕\\Time_Series_Instruct-main\\UCRArchive_2018\\{task_name}\\{task_name}_{mode}.tsv", "r", encoding="utf-8") as f:
        for row in f:
            row = row.strip()
            if not row:
                continue

            label_ = row.split("\t")[0]
            label_name_ = label_map[label_]
            label_zh_name_ = label_map_zh[label_]

            ts_data_ = row.split("\t")[1: ]
            assert len(ts_data_) == 463
            ts_data_blank_sep = " ".join(ts_data_)
            ts_data_comma_sep = ",".join(ts_data_)

            for template_ in list_templates:
                template_ = copy.copy(template_)
                # print(template_)

                # 确认数据怎么分割的
                if "<ts-data-comma-sep>" in template_:
                    template_ = template_.replace(
                        r"<ts-data-comma-sep>",
                        ts_data_comma_sep
                    )
                else:
                    template_ = template_.replace(
                        r"<ts-data-blank-sep>",
                        ts_data_blank_sep
                    )

                assert "<ts-data-comma-sep>" not in template_
                assert "<ts-data-blank-sep>" not in template_

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
