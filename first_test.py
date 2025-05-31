import plotly.express as px  # 用于绘图
import streamlit as st
import pandas as pd
import sqlite3
import os
import re  # 用于 extract_boss_details

# --- 0. 配置项和辅助函数 ---
COLUMN_MAPPINGS = {
    "battlefield_level": "params_stageLv",
    "battlefield_name": "stageName",
    "report_id": "reportId",
    "entity_name_final": "entity_name_final",
    "is_boss": "entity_is_boss",  # 确保你的数据库列名是 entity_is_boss
    "star_level": "entity_plunderStar",
    "ore_mining": "entity_explois",
    "kill_count": "entity_killCount",
    "death_count": "entity_dieCount",
    "entity_id": "entity_id"
}
BATTLEFIELD_TOTAL_ORE = {
    70: 6376380,
    80: 9468768,
    90: 11266616,
    100: 12666778,
    110: 16492170,
    120: 20501074,
    130: 28831204,
    140: 40459070,
    150: 54252688,
    160: 67756400,
    170: 85814462,
    180: 108943684,
    190: 127740848,
    200: 149781280
}

def get_col(internal_name):
    return COLUMN_MAPPINGS.get(internal_name, internal_name)


DISPLAY_NAMES_MAP = {
    get_col("battlefield_level"): "战场等级",
    get_col("reportId"): "战斗ID",
    get_col("entity_name_final"): "角色名",
    get_col("is_boss"): "是否Boss",
    get_col("star_level"): "星级",
    get_col("ore_mining"): "拆矿量",
    get_col("kill_count"): "击杀数",
    get_col("death_count"): "死亡数",
    get_col("entity_id"): "实体记录ID",
    get_col("costTime"): "消耗时间(秒)",
    # get_col("fightVer"): "原始版本号", # 你可能不想直接显示原始版本号
    get_col("time"): "原始时间戳",  # 同上
    'major_fight_ver_display': "游戏版本",
    'datetime': "发生日期时间",
    'one_life_ore_mining': "一命拆矿效率",
    'boss_base_name': "魔王",
    'boss_tier': "魔王阶位",
    'boss_fullname_with_tier': "魔王全称(分析用)",  # 这个可能不需要直接展示给用户
    'avg_hero_star': "匹配神将平均星级",

    # Tab1 生成的列
    '出现次数 (独立报告ID)': "报告ID计数",
    '占筛选后总报告ID百分比 (%)': "占总报告ID百分比(%)",

    # Tab2 生成的列
    '角色名称 (含阶)': "角色名(统计)",  # 沿用你之前的
    '出现次数': "总出现次数",
    '占筛选后总实体出现次数百分比 (%)': "占总实体出现百分比(%)",

    # Tab5 生成的列
    '最常匹配魔王': '最常匹配魔王',
    '匹配次数': '与该星级神将匹配次数',
    '该星级神将总匹配数': '该星级神将总战斗数',
    '占比 (%)': '匹配占比(%)',

    # 你可能还需要为 pivot_table 的索引名添加映射，如果它们被 reset_index() 变成了列
    # 比如，如果 pivot_table 的 index 是 'entity_name_final'，reset_index 后它就成了列
}


# 对于数值型的列名 (如pivot_table的columns是0,1,2,3...)
# 我们在显示前动态添加后缀，如 "0星", "1星"

def rename_df_columns_for_display(df, display_map):
    if df is None or df.empty:
        return df

    current_cols = df.columns.tolist()
    rename_dict = {}
    new_cols_order = []  # 用于保持原始顺序或自定义顺序

    for col in current_cols:
        if col in display_map:
            new_name = display_map[col]
            rename_dict[col] = new_name
            new_cols_order.append(new_name)
        elif isinstance(col, (int, float)):  # 处理数值型列名, 如星级/阶数
            # 假设这里我们想给星级/阶数加上 "星" 或 "阶"
            # 这需要上下文判断当前DataFrame是什么。
            # 为了通用性，这个函数可能不适合处理这种动态后缀。
            # 这种后缀最好在生成DataFrame后，在特定Tab内处理。
            # 或者，如果约定所有数字列都是星级，可以简单处理：
            # new_name = f"{col}星"
            # rename_dict[col] = new_name
            # new_cols_order.append(new_name)
            # 但更好的方式是在各个tab内对pivot_table结果的列单独处理
            new_cols_order.append(col)  # 保持原样
        else:
            new_cols_order.append(col)  # 保持原样

    df_renamed = df.rename(columns=rename_dict)

    # 如果 DataFrame 的索引有名字，也尝试重命名
    # if df_renamed.index.name and df_renamed.index.name in display_map:
    #     df_renamed = df_renamed.rename_axis(display_map[df_renamed.index.name])

    return df_renamed  # [new_cols_order] # 暂时不强制重排，rename已足够
def extract_boss_details(entity_name_final_col):
    if not isinstance(entity_name_final_col, pd.Series):
        if isinstance(entity_name_final_col, str):
            match = re.match(r'(.+?)(\d+)阶$', entity_name_final_col)
            if match: return match.group(1), int(match.group(2))
            return entity_name_final_col, 0
        return pd.Series([None, 0], index=['boss_base_name', 'boss_tier'])

    def extractor(name):
        if not isinstance(name, str): return name, 0
        match = re.match(r'(.+?)(\d+)阶$', name)
        if match: return match.group(1), int(match.group(2))
        return name, 0

    extracted_data = entity_name_final_col.apply(extractor)
    return pd.DataFrame(extracted_data.tolist(), index=entity_name_final_col.index,
                        columns=['boss_base_name', 'boss_tier'])


# --- 1. 定义数据库文件路径 ---
current_dir = os.path.dirname(__file__)
db_file_path = os.path.join(current_dir, 'data.db')
table_name = 'data'


# --- 2. 定义数据加载函数并使用缓存 ---
@st.cache_data
def load_data(db_path, table):
    print(f"正在加载数据从 {db_path} 的表 {table}...")
    try:
        conn = sqlite3.connect(db_path)
        query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        numeric_cols_internal = ["star_level", "ore_mining", "kill_count", "death_count", "battlefield_level",
                                 "entity_id"]
        numeric_cols_internal_df = ['score', 'costTime'] + [get_col(name) for name in numeric_cols_internal]
        for col_name_in_df in numeric_cols_internal_df:
            if col_name_in_df in df.columns:
                df[col_name_in_df] = pd.to_numeric(df[col_name_in_df], errors='coerce').fillna(0)
                if col_name_in_df in [get_col("battlefield_level"), get_col("star_level"), get_col("kill_count"),
                                      get_col("death_count"), get_col("entity_id"), 'score', 'costTime']:
                    df[col_name_in_df] = df[col_name_in_df].astype(int)

        is_boss_col_name = get_col("is_boss")
        if is_boss_col_name in df.columns:
            if df[is_boss_col_name].dtype == 'object':
                df[is_boss_col_name] = df[is_boss_col_name].astype(str).str.upper().map(
                    {'TRUE': True, 'FALSE': False, '1': True, '0': False}).fillna(False)
            elif pd.api.types.is_numeric_dtype(df[is_boss_col_name]):
                df[is_boss_col_name] = df[is_boss_col_name].astype(bool)

        for col_name in [get_col("report_id"), get_col("entity_name_final"), get_col("battlefield_name")]:
            if col_name in df.columns:
                df[col_name] = df[col_name].astype(str)
        fight_ver_col = get_col('fightVer')  # 假设你的 COLUMN_MAPPINGS 中有 fightVer
        if fight_ver_col not in COLUMN_MAPPINGS.values():  # 如果 fightVer 不在映射中，直接使用 'fightVer'
            fight_ver_col = 'fightVer'  # 或者你数据库中实际的列名

        if fight_ver_col in df.columns:
            # 确保 fightVer 是字符串以便进行切片，如果已经是数值，先转字符串
            df[fight_ver_col] = df[fight_ver_col].astype(str)
            # 提取前3位作为大版本号 (例如 "22504" -> "225")
            df['major_fight_ver_raw'] = df[fight_ver_col].str[:3]
            # 将大版本号转换为 "X.XX" 格式 (例如 "225" -> 2.25)
            # errors='coerce' 会将无法转换的设为 NaN
            df['major_fight_ver_display'] = pd.to_numeric(df['major_fight_ver_raw'], errors='coerce') / 100
            # 对于无法转换的 (NaN)，可以用一个特殊值代替或保留 NaN
            # df['major_fight_ver_display'] = df['major_fight_ver_display'].fillna("未知版本")
        else:
            print(f"警告: 列 '{fight_ver_col}' 在数据中未找到，无法进行版本号筛选。")
            # 创建空列以避免后续代码出错
            df['major_fight_ver_raw'] = pd.Series(dtype='str')
            df['major_fight_ver_display'] = pd.Series(dtype='float')
        print("数据加载和类型转换完成。")


        time_col = get_col('time')  # 假设你的 COLUMN_MAPPINGS 中有 time
        if time_col not in COLUMN_MAPPINGS.values():
            time_col = 'time'  # 或者你数据库中实际的列名

        if time_col in df.columns:
            # 你的时间戳是 E+12 格式，通常是毫秒级时间戳
            # pd.to_datetime 需要数值型，先确保它是数值
            df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            df['datetime'] = pd.to_datetime(df[time_col], unit='ms', errors='coerce')
            # errors='coerce' 会将无法转换的设为 NaT (Not a Time)
        else:
            print(f"警告: 列 '{time_col}' 在数据中未找到，无法进行时间筛选。")
            df['datetime'] = pd.Series(dtype='datetime64[ns]')

        return df
    except Exception as e:
        st.error(f"加载或处理数据时发生错误: {e}")
        print(f"加载数据时发生错误: {e}")
        return pd.DataFrame()


# --- 3. 调用数据加载函数 ---
df_original = load_data(db_file_path, table_name)

# --- 4. Streamlit 页面内容 ---
# st.set_page_config(layout="wide") # 如果你的Streamlit版本支持，可以取消注释
st.title("zmws神魔数据")

if df_original.empty:
    st.warning("未能加载到数据，请检查数据库文件、表名或数据加载函数。")
    st.stop()

# --- 5. 添加筛选器 ---
st.sidebar.header('数据筛选')

show_boss_options = ['双方数据', '神将数据', '魔王数据']
selected_show_boss_type = st.sidebar.selectbox('选择查看方', show_boss_options)

all_stage_levels_options = ['所有等级'] + sorted(df_original[get_col("battlefield_level")].unique().tolist())
selected_stage_level = st.sidebar.selectbox('选择战场等级', all_stage_levels_options)

all_star_levels_options = ['所有星级'] + sorted(df_original[get_col("star_level")].unique().tolist())
selected_star_level = st.sidebar.selectbox('选择神将星级', all_star_levels_options)
st.sidebar.markdown("---") # 添加分隔线，区分不同类型的筛选器
st.sidebar.write("**高级筛选器**")

# 筛选器: 消耗时间 (costTime)
# 获取 costTime 的最小值和最大值作为滑块的范围
min_cost_time = int(df_original[get_col('costTime')].min())
max_cost_time = int(df_original[get_col('costTime')].max())

if min_cost_time < max_cost_time : # 只有当范围有效时才显示滑块
    selected_cost_time_range = st.sidebar.slider(
        '选择消耗时间范围 (秒)',
        min_value=min_cost_time,
        max_value=max_cost_time,
        value=(min_cost_time, max_cost_time), # 默认选择整个范围
        key="cost_time_slider"
    )
else: # 如果所有 costTime 都一样，或数据为空
    st.sidebar.caption(f"消耗时间数据单一 ({min_cost_time}秒)，无需筛选。")
    selected_cost_time_range = (min_cost_time, max_cost_time) # 保持一个默认值
if 'major_fight_ver_display' in df_original.columns:
    # 获取所有唯一的大版本号 (X.XX 格式)，去除 NaN，排序
    major_versions = sorted(df_original['major_fight_ver_display'].dropna().unique().tolist())
    if major_versions: # 只有当有有效版本号时才显示筛选器
        selected_major_versions = st.sidebar.multiselect(
            '选择游戏版本 (可多选)',
            options=major_versions,
            default=[], # 默认不选择任何版本 (即显示所有版本的数据)
            format_func=lambda x: f"{x:.2f}", # 确保选项显示为两位小数
            key="major_ver_multiselect"
        )
    else:
        st.sidebar.caption("无有效游戏版本数据可供筛选。")
        selected_major_versions = [] # 保持默认值
else:
    selected_major_versions = [] # 如果列不存在，则无筛选

if 'datetime' in df_original.columns and not df_original['datetime'].dropna().empty:
    st.sidebar.markdown("---")
    st.sidebar.write("**时间筛选器**")

    min_date = df_original['datetime'].dropna().min().date()
    max_date = df_original['datetime'].dropna().max().date()

    time_filter_mode = st.sidebar.selectbox(
        "选择时间筛选模式:",
        options=["不筛选", "日期范围", "某个日期之前", "某个日期之后"],
        key="time_mode_select"
    )

    selected_start_date = None
    selected_end_date = None

    if time_filter_mode == "日期范围":
        # Streamlit 的 st.date_input 返回的是 datetime.date 对象
        selected_start_date = st.sidebar.date_input('开始日期', value=min_date, min_value=min_date, max_value=max_date, key="start_date")
        selected_end_date = st.sidebar.date_input('结束日期', value=max_date, min_value=min_date, max_value=max_date, key="end_date")
        if selected_start_date > selected_end_date:
            st.sidebar.error("开始日期不能晚于结束日期！")
            # 阻止进一步的筛选或使用默认值
            selected_start_date, selected_end_date = min_date, max_date


    elif time_filter_mode == "某个日期之前":
        selected_end_date = st.sidebar.date_input('选择截止日期 (包含当天)', value=max_date, min_value=min_date, max_value=max_date, key="before_date")
        # 开始日期可以认为是数据中的最早日期或不限制
        selected_start_date = None # 表示从最早开始

    elif time_filter_mode == "某个日期之后":
        selected_start_date = st.sidebar.date_input('选择起始日期 (包含当天)', value=min_date, min_value=min_date, max_value=max_date, key="after_date")
        # 结束日期可以认为是数据中的最晚日期或不限制
        selected_end_date = None # 表示到最晚结束

    # 工作日/周末筛选 (作为一个独立的筛选器，可以与其他时间筛选模式组合)
    weekday_weekend_filter = st.sidebar.multiselect(
        "筛选工作日/周末 (可多选):",
        options=["工作日", "周末"],
        default=[], # 默认不应用此筛选
        key="weekday_weekend_select"
    )

else: # 如果没有有效的 datetime 数据
    time_filter_mode = "不筛选"
    weekday_weekend_filter = []
# --- 6. 应用筛选逻辑 ---
filtered_df = df_original.copy()

if selected_show_boss_type == '神将数据':
    filtered_df = filtered_df[filtered_df[get_col("is_boss")] == False]
elif selected_show_boss_type == '魔王数据':
    filtered_df = filtered_df[filtered_df[get_col("is_boss")] == True]

if selected_stage_level != '所有等级':
    filtered_df = filtered_df[filtered_df[get_col("battlefield_level")] == selected_stage_level]

if selected_star_level != '所有星级':
    if selected_show_boss_type != '魔王数据':
        filtered_df = filtered_df[filtered_df[get_col("star_level")] == selected_star_level]
if min_cost_time < max_cost_time : # 确保滑块被正确初始化
    filtered_df = filtered_df[
        (filtered_df[get_col('costTime')] >= selected_cost_time_range[0]) &
        (filtered_df[get_col('costTime')] <= selected_cost_time_range[1])
    ]
if selected_major_versions: # 只有当用户选择了版本时才应用筛选
    # 我们需要基于 'major_fight_ver_display' 列进行筛选
    filtered_df = filtered_df[filtered_df['major_fight_ver_display'].isin(selected_major_versions)]
if 'datetime' in filtered_df.columns and time_filter_mode != "不筛选":
    # 先将 Pandas Series 中的 NaT (Not a Time) 去掉，避免比较错误
    valid_datetime_series = filtered_df['datetime'].dropna()

    if time_filter_mode == "日期范围" and selected_start_date and selected_end_date:
        # 将 selected_start_date 和 selected_end_date 转换为 datetime64 以便比较
        # .dt.date 用于比较日期部分，忽略时间
        start_datetime = pd.to_datetime(selected_start_date)
        end_datetime = pd.to_datetime(selected_end_date) + pd.Timedelta(days=1)  # 结束日期包含当天，所以到下一天的0点

        filtered_df = filtered_df[
            (filtered_df['datetime'] >= start_datetime) &
            (filtered_df['datetime'] < end_datetime)  # 小于下一天0点
            ]
    elif time_filter_mode == "某个日期之前" and selected_end_date:
        end_datetime = pd.to_datetime(selected_end_date) + pd.Timedelta(days=1)
        filtered_df = filtered_df[filtered_df['datetime'] < end_datetime]

    elif time_filter_mode == "某个日期之后" and selected_start_date:
        start_datetime = pd.to_datetime(selected_start_date)
        filtered_df = filtered_df[filtered_df['datetime'] >= start_datetime]

# 应用工作日/周末筛选
if 'datetime' in filtered_df.columns and weekday_weekend_filter:
    # .dt.dayofweek: Monday=0, Sunday=6
    is_weekday = filtered_df['datetime'].dt.dayofweek < 5  # 0-4 是工作日
    is_weekend = filtered_df['datetime'].dt.dayofweek >= 5  # 5-6 是周末

    conditions_to_keep = pd.Series([False] * len(filtered_df), index=filtered_df.index)  # 初始化为全False

    if "工作日" in weekday_weekend_filter:
        conditions_to_keep = conditions_to_keep | is_weekday
    if "周末" in weekday_weekend_filter:
        conditions_to_keep = conditions_to_keep | is_weekend

    if conditions_to_keep.any():  # 只有当有条件为True时才应用筛选
        filtered_df = filtered_df[conditions_to_keep]
    elif weekday_weekend_filter:  # 如果选了但没有匹配（例如只选工作日，但数据全是周末），则结果为空
        filtered_df = filtered_df.iloc[0:0]  # 返回空DataFrame


# --- 7. 使用 Tabs 展示不同的统计信息 ---
st.header("详细统计分析")
tab_titles = ["战场等级次数", "角色出现次数", "神将星级战场次数", "神将星级战场表现", "魔王阶数战场表现","匹配分析"]
tabs = st.tabs(tab_titles)

# ... (假设你的脚本的其他部分，如 import, COLUMN_MAPPINGS, 辅助函数, load_data,
#      df_original 的加载, 侧边栏筛选器定义, filtered_df 的计算,
#      以及 st.tabs 的定义都保持不变) ...

# 确保你的 tabs 变量是这样定义的 (或类似，索引要对应)：
# tab_titles = [
#     "战场等级次数", "角色出现次数", "神将星级战场次数",
#     "神将星级战场表现", "魔王阶数战场表现", "匹配分析",
#     "探索性分析", "筛选后原始数据"
# ]
# tabs = st.tabs(tab_titles)

with tabs[0]:  # 战场等级次数 Tab
    st.subheader("1. 各等级战场出现次数")
    if not filtered_df.empty:
        df_s1_counts = filtered_df.groupby(get_col("battlefield_level"))[get_col("report_id")].nunique().reset_index()
        df_s1_counts.columns = [get_col("battlefield_level"), '出现次数 (独立报告ID)']

        if not df_s1_counts.empty:
            total_reports_in_filtered_df = filtered_df[get_col("report_id")].nunique()
            df_s1_counts = df_s1_counts.sort_values(by='出现次数 (独立报告ID)', ascending=False)

            fig_s1 = px.bar(df_s1_counts,
                            x=get_col("battlefield_level"),
                            y='出现次数 (独立报告ID)',
                            title='各等级战场出现次数',
                            labels={get_col("battlefield_level"): "战场等级"})
            fig_s1.update_xaxes(type='category')
            st.plotly_chart(fig_s1, use_container_width=True)
            st.caption(f"当前筛选条件下，总独立报告ID数量: {total_reports_in_filtered_df}")
        else:
            st.info("当前筛选条件下无数据可用于此统计。")
    else:
        st.info("请先应用筛选条件以查看统计。")

with tabs[1]:  # 角色出现次数 Tab
    st.subheader("2. 各角色出现次数")
    if not filtered_df.empty:
        df_s2_counts = filtered_df[get_col("entity_name_final")].value_counts().reset_index()
        df_s2_counts.columns = ['角色名称 (含阶)', '出现次数']
        if not df_s2_counts.empty:
            total_entity_occurrences_in_filtered_df = len(filtered_df)
            df_s2_counts = df_s2_counts.sort_values(by='出现次数', ascending=False)

            fig_s2 = px.bar(df_s2_counts.head(30),
                            x='角色名称 (含阶)',
                            y='出现次数',
                            title='各角色出现次数 (Top 30)')
            st.plotly_chart(fig_s2, use_container_width=True)
            st.caption(f"当前筛选条件下，总实体记录行数: {total_entity_occurrences_in_filtered_df}")
        else:
            st.info("当前筛选条件下无数据可用于此统计。")
    else:
        st.info("请先应用筛选条件以查看统计。")

with tabs[2]:  # 神将星级战场次数 Tab
    st.subheader("3. 神将各星级出现次数 (样本量)")
    df_for_tab3 = filtered_df.copy()
    if selected_show_boss_type == '魔王数据':
        st.info("此统计项仅适用于神将数据。请在侧边栏选择“神将数据”或“双方数据”。")
    elif selected_show_boss_type == '双方数据':
        df_for_tab3 = df_for_tab3[df_for_tab3[get_col("is_boss")] == False]

    if not df_for_tab3.empty:
        pivot_title_suffix = ""
        if selected_stage_level == '所有等级':
            st.write("当前显示: **所有战场等级汇总** 的神将星级出现次数")
            pivot_title_suffix = " (所有战场等级汇总)"
            df_to_pivot = df_for_tab3
        else:
            st.write(f"当前显示: 战场等级 **{selected_stage_level}** 的神将星级出现次数")
            pivot_title_suffix = f" (战场等级 {selected_stage_level})"
            df_to_pivot = df_for_tab3
        try:
            pivot_s3 = pd.pivot_table(df_to_pivot,
                                      values=get_col("entity_id"),
                                      index=get_col("entity_name_final"),
                                      columns=get_col("star_level"),
                                      aggfunc='count',
                                      fill_value=0)
            if not pivot_s3.empty:
                # 根据侧边栏的 selected_star_level 决定热力图的数据 (只影响显示，不影响计算)
                data_for_heatmap_s3 = pivot_s3.copy()
                title_for_heatmap_s3 = f"神将各星级出现次数{pivot_title_suffix}"

                if selected_star_level != '所有星级':
                    if selected_star_level in data_for_heatmap_s3.columns:
                        data_for_heatmap_s3 = data_for_heatmap_s3[[selected_star_level]]
                        title_for_heatmap_s3 = f"神将 {selected_star_level}星 出现次数{pivot_title_suffix}"
                    else:
                        st.caption(f"提示: 选择的星级 '{selected_star_level}' 在当前数据中不存在，热力图显示所有星级。")

                if not data_for_heatmap_s3.empty and len(data_for_heatmap_s3.columns) > 0:
                    fig_s3 = px.imshow(data_for_heatmap_s3,
                                       title=title_for_heatmap_s3,
                                       text_auto=True,
                                       aspect="auto",
                                       labels=dict(x="神将星级", y="神将名称", color="出现次数"),
                                       color_continuous_scale="Viridis")
                    fig_s3.update_xaxes(type='category')
                    fig_s3.update_yaxes(type='category')
                    st.plotly_chart(fig_s3, use_container_width=True)
                else:
                    st.info(f"当前筛选条件下，没有足够的星级数据为“神将各星级出现次数”生成热力图{pivot_title_suffix}。")
            else:
                st.info(f"当前筛选条件下无神将数据可用于此统计{pivot_title_suffix}。")
        except Exception as e:
            st.error(f"计算透视表时发生错误: {e}")
    elif selected_show_boss_type != '魔王数据':
        st.info("当前筛选条件下无神将数据可用于此统计。")

with tabs[3]:  # 神将星级战场表现 Tab
    st.subheader("4. 神将各星级在不同战场等级的表现")
    df_for_tab4 = filtered_df.copy()
    if selected_show_boss_type == '魔王数据':
        st.info("此统计项仅适用于神将数据。请在侧边栏选择“神将数据”或“双方数据”。")
    elif selected_show_boss_type == '双方数据':
        df_for_tab4 = df_for_tab4[df_for_tab4[get_col("is_boss")] == False]

    if not df_for_tab4.empty:
        df_for_tab4['one_life_ore_mining'] = (df_for_tab4[get_col("ore_mining")] /
                                              (df_for_tab4[get_col("death_count")] + 1)).fillna(0)
        current_battlefield_total_ore = None
        pivot_context_title_suffix = ""
        if selected_stage_level == '所有等级':
            st.write("当前显示: **所有战场等级汇总** 的神将星级表现")
            pivot_context_title_suffix = " (所有战场等级汇总)"
            df_to_analyze = df_for_tab4
        else:
            st.write(f"当前显示: 战场等级 **{selected_stage_level}** 的神将星级表现")
            pivot_context_title_suffix = f" (战场等级 {selected_stage_level})"
            df_to_analyze = df_for_tab4
            current_battlefield_total_ore = BATTLEFIELD_TOTAL_ORE.get(selected_stage_level, None)
            if current_battlefield_total_ore is None and selected_stage_level != '所有等级':
                st.warning(f"未找到战场等级 {selected_stage_level} 的总矿量数据，百分比视图将不可用。")

        metrics_to_pivot = {
            get_col("ore_mining"): {'aggfunc': 'mean', 'title_suffix': '平均拆矿量', 'color_scale': 'Greens',
                                    'format': ".0f", 'can_show_percentage': True},
            'one_life_ore_mining': {'aggfunc': 'mean', 'title_suffix': '平均一命拆矿量', 'color_scale': 'Purples',
                                    'format': ".0f", 'can_show_percentage': True},
            get_col("death_count"): {'aggfunc': 'mean', 'title_suffix': '平均死亡次数', 'color_scale': 'Reds',
                                     'format': ".2f"},
            get_col("kill_count"): {'aggfunc': 'mean', 'title_suffix': '平均击杀次数', 'color_scale': 'Blues',
                                    'format': ".2f"},
            get_col("entity_id"): {'aggfunc': 'count', 'title_suffix': '出现次数 (样本量)', 'color_scale': 'Oranges',
                                   'format': ".0f"}
        }
        for value_col_key, config in metrics_to_pivot.items():
            actual_value_col = value_col_key if value_col_key == 'one_life_ore_mining' else get_col(value_col_key)
            if actual_value_col not in df_to_analyze.columns:
                st.warning(f"指标列 '{actual_value_col}' (用于 '{config['title_suffix']}') 在数据中不存在，跳过此统计。")
                continue
            st.markdown(f"#### {config['title_suffix']}{pivot_context_title_suffix}")
            view_type = "绝对值"
            if config.get(
                    'can_show_percentage') and current_battlefield_total_ore and current_battlefield_total_ore > 0:
                view_type = st.radio("选择视图:", ("绝对值", "占战场总矿量百分比"), key=f"view_radio_{value_col_key}",
                                     horizontal=True)
            try:
                pivot_table_abs = pd.pivot_table(df_to_analyze, values=actual_value_col,
                                                 index=get_col("entity_name_final"), columns=get_col("star_level"),
                                                 aggfunc=config['aggfunc'], fill_value=0)
                if not pivot_table_abs.empty:
                    data_to_display_in_chart = pivot_table_abs
                    current_format = config['format']
                    current_color_scale = config['color_scale']
                    chart_title = f"{config['title_suffix']}{pivot_context_title_suffix}"
                    color_bar_label = config['title_suffix']
                    if view_type == "占战场总矿量百分比":
                        pivot_table_perc = (pivot_table_abs / current_battlefield_total_ore * 100)
                        data_to_display_in_chart = pivot_table_perc
                        current_format = ".2f"
                        chart_title = f"{config['title_suffix']} (占总矿量 %){pivot_context_title_suffix}"
                        color_bar_label = f"{config['title_suffix']} (%)"

                    # 根据侧边栏的 selected_star_level 筛选图表数据列
                    data_for_heatmap_tab4 = data_to_display_in_chart.copy()
                    title_for_heatmap_tab4 = chart_title
                    if selected_star_level != '所有星级':
                        if selected_star_level in data_for_heatmap_tab4.columns:
                            data_for_heatmap_tab4 = data_for_heatmap_tab4[[selected_star_level]]
                            # 更新标题以反映只显示特定星级
                            star_suffix = f" ({selected_star_level}星)"
                            if view_type == "占战场总矿量百分比":
                                title_for_heatmap_tab4 = f"{config['title_suffix']} (占总矿量 %){star_suffix}{pivot_context_title_suffix}"
                            else:
                                title_for_heatmap_tab4 = f"{config['title_suffix']}{star_suffix}{pivot_context_title_suffix}"
                        else:
                            st.caption(f"提示: 选择的星级 '{selected_star_level}' 在当前数据中不存在，图表显示所有星级。")

                    if len(data_for_heatmap_tab4.columns) > 0:
                        hovertemplate_dynamic = (
                            f"<b>神将: %{{y}}</b><br>星级: %{{x}}<br>{'百分比' if view_type == '占战场总矿量百分比' else config['title_suffix']}: %{{z:{current_format}}}{'%' if view_type == '占战场总矿量百分比' else ''}<br><extra></extra>")
                        fig = px.imshow(data_for_heatmap_tab4, title=title_for_heatmap_tab4,
                                        color_continuous_scale=current_color_scale, aspect="auto",
                                        labels=dict(x="神将星级", y="神将名称", color=color_bar_label))
                        fig.update_traces(
                            texttemplate=f"%{{z:{current_format}}}{'%' if view_type == '占战场总矿量百分比' else ''}",
                            hovertemplate=hovertemplate_dynamic)
                        fig.update_xaxes(type='category')
                        fig.update_yaxes(type='category')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"当前筛选条件下，没有足够的星级数据来为“{config['title_suffix']}”生成图表。")
                else:
                    st.info(f"当前筛选条件下无神将数据可用于计算“{config['title_suffix']}”。")
            except Exception as e:
                st.error(f"计算或绘制“{config['title_suffix']}”图表时发生错误: {e}")
            st.write("---")
    elif selected_show_boss_type != '魔王数据':
        st.info("当前筛选条件下无神将数据可用于此统计。")

with tabs[4]:  # 魔王阶数战场表现 Tab
    st.subheader("5. 魔王各阶在不同战场等级的表现")
    df_for_tab5 = filtered_df.copy()
    if selected_show_boss_type == '神将数据':
        st.info("此统计项仅适用于魔王数据。请在侧边栏选择“魔王数据”或“双方数据”。")
    elif selected_show_boss_type == '双方数据':
        df_for_tab5 = df_for_tab5[df_for_tab5[get_col("is_boss")] == True]

    if not df_for_tab5.empty:
        try:
            boss_details_df = extract_boss_details(df_for_tab5[get_col("entity_name_final")])
            df_for_tab5 = df_for_tab5.assign(boss_base_name=boss_details_df['boss_base_name'],
                                             boss_tier=pd.to_numeric(boss_details_df['boss_tier'],
                                                                     errors='coerce').fillna(0).astype(int))
        except Exception as e:
            st.error(f"提取Boss详情时发生错误: {e}")
            df_for_tab5 = pd.DataFrame()

    if not df_for_tab5.empty and 'boss_base_name' in df_for_tab5.columns:  # 确保 boss 详情已成功添加
        pivot_context_title_suffix = ""
        if selected_stage_level == '所有等级':
            st.write("当前显示: **所有战场等级汇总** 的魔王各阶表现")
            pivot_context_title_suffix = " (所有战场等级汇总)"
            df_to_analyze_boss = df_for_tab5
        else:
            st.write(f"当前显示: 战场等级 **{selected_stage_level}** 的魔王各阶表现")
            pivot_context_title_suffix = f" (战场等级 {selected_stage_level})"
            df_to_analyze_boss = df_for_tab5

        boss_metrics_to_pivot = {
            get_col("kill_count"): {'aggfunc': 'mean', 'title_suffix': '平均击杀次数', 'color_scale': 'Greens',
                                    'format': ".2f"},
            get_col("death_count"): {'aggfunc': 'mean', 'title_suffix': '平均死亡次数', 'color_scale': 'Reds',
                                     'format': ".2f"},
            get_col("entity_id"): {'aggfunc': 'count', 'title_suffix': '出现次数 (样本量)', 'color_scale': 'Oranges',
                                   'format': ".0f"}
        }
        for value_col, config in boss_metrics_to_pivot.items():
            st.markdown(f"#### {config['title_suffix']}{pivot_context_title_suffix}")
            try:
                pivot_table_boss = pd.pivot_table(df_to_analyze_boss, values=value_col, index='boss_base_name',
                                                  columns='boss_tier', aggfunc=config['aggfunc'], fill_value=0)
                if not pivot_table_boss.empty and len(pivot_table_boss.columns) > 0:
                    fig_boss = px.imshow(pivot_table_boss,
                                         title=f"{config['title_suffix']}{pivot_context_title_suffix}",
                                         text_auto=config['format'], aspect="auto",
                                         labels=dict(x="魔王阶数", y="魔王名称", color=config['title_suffix']),
                                         color_continuous_scale=config['color_scale'])
                    fig_boss.update_xaxes(type='category')
                    fig_boss.update_yaxes(type='category')
                    st.plotly_chart(fig_boss, use_container_width=True)
                elif not pivot_table_boss.empty and len(pivot_table_boss.columns) == 0:
                    st.info(f"当前筛选条件下，没有足够的魔王阶数数据为“{config['title_suffix']}”生成热力图。")
                else:
                    st.info(f"当前筛选条件下无魔王数据可用于计算“{config['title_suffix']}”。")
            except KeyError as e:
                st.error(f"计算“{config['title_suffix']}”时发生列名错误: {e}。")
            except Exception as e:
                st.error(f"计算或绘制“{config['title_suffix']}”图表时发生错误: {e}")
            st.write("---")
    elif selected_show_boss_type != '神将数据':
        st.info("当前筛选条件下无魔王数据可用于此统计。")

with tabs[5]:  # "匹配分析" Tab
    st.subheader("匹配机制分析")
    df_for_match_analysis_base = df_original.copy()
    if selected_stage_level != '所有等级':
        df_for_match_analysis_base = df_for_match_analysis_base[
            df_for_match_analysis_base[get_col("battlefield_level")] == selected_stage_level]

    st.markdown("##### **Tab 内筛选器**")
    # ... (Tab内筛选器定义) ...
    temp_bosses_for_options = df_for_match_analysis_base[df_for_match_analysis_base[get_col("is_boss")] == True].copy()
    all_boss_base_names_options = []
    all_boss_tiers_options = []
    if not temp_bosses_for_options.empty:
        try:
            boss_details_options = extract_boss_details(temp_bosses_for_options[get_col("entity_name_final")])
            all_boss_base_names_options = sorted(boss_details_options['boss_base_name'].dropna().unique().tolist())
            all_boss_tiers_options = sorted(
                pd.to_numeric(boss_details_options['boss_tier'], errors='coerce').dropna().unique().astype(
                    int).tolist())
        except Exception as e:
            st.warning(f"为Tab内筛选器准备魔王选项时出错: {e}")
    selected_boss_names_tab = st.multiselect("选择特定魔王 (基础名称):", options=all_boss_base_names_options,
                                             default=[], key="match_tab_boss_name_select")
    selected_boss_tiers_tab = st.multiselect("选择特定魔王阶数:", options=all_boss_tiers_options, default=[],
                                             key="match_tab_boss_tier_select")
    tab_filter_conditions = []
    if selected_boss_names_tab: tab_filter_conditions.append(f"魔王名: {', '.join(selected_boss_names_tab)}")
    if selected_boss_tiers_tab: tab_filter_conditions.append(
        f"魔王阶数: {', '.join(map(str, selected_boss_tiers_tab))}")
    dynamic_title_suffix = " (筛选条件: " + "; ".join(tab_filter_conditions) + ")" if tab_filter_conditions else ""
    st.caption(
        f"提示: Tab内筛选器作用于本页分析。全局战场等级: {selected_stage_level if selected_stage_level != '所有等级' else '所有'}。")
    st.markdown("---")

    heroes_df_tab_base = df_for_match_analysis_base[df_for_match_analysis_base[get_col("is_boss")] == False].copy()
    bosses_df_tab_base = df_for_match_analysis_base[df_for_match_analysis_base[get_col("is_boss")] == True].copy()
    final_bosses_df_for_tab = bosses_df_tab_base.copy()
    if not final_bosses_df_for_tab.empty:
        try:
            boss_details_for_filter = extract_boss_details(final_bosses_df_for_tab[get_col("entity_name_final")])
            final_bosses_df_for_tab = final_bosses_df_for_tab.assign(
                boss_base_name_temp=boss_details_for_filter['boss_base_name'],
                boss_tier_temp=pd.to_numeric(boss_details_for_filter['boss_tier'], errors='coerce').fillna(-1).astype(
                    int))
            if selected_boss_names_tab: final_bosses_df_for_tab = final_bosses_df_for_tab[
                final_bosses_df_for_tab['boss_base_name_temp'].isin(selected_boss_names_tab)]
            if selected_boss_tiers_tab: final_bosses_df_for_tab = final_bosses_df_for_tab[
                final_bosses_df_for_tab['boss_tier_temp'].isin(selected_boss_tiers_tab)]
            final_bosses_df_for_tab = final_bosses_df_for_tab.drop(columns=['boss_base_name_temp', 'boss_tier_temp'],
                                                                   errors='ignore')
        except Exception as e:
            st.error(f"应用Tab内魔王筛选时发生错误: {e}")
            final_bosses_df_for_tab = pd.DataFrame()

    current_bosses_df = final_bosses_df_for_tab
    current_heroes_df = heroes_df_tab_base
    current_bosses_df_an1 = pd.DataFrame()  # Initialize to handle potential errors in an1
    current_bosses_df_an2 = pd.DataFrame()  # Initialize

    st.markdown("#### 1. 每个魔王匹配的神将平均星级")
    if current_bosses_df.empty or current_heroes_df.empty:
        st.info("需要有效的魔王和神将数据才能进行分析一。")
    else:
        try:
            boss_details_an1 = extract_boss_details(current_bosses_df[get_col("entity_name_final")])
            current_bosses_df_an1 = current_bosses_df.assign(boss_base_name=boss_details_an1['boss_base_name'],
                                                             boss_tier=pd.to_numeric(boss_details_an1['boss_tier'],
                                                                                     errors='coerce').fillna(0).astype(
                                                                 int))
            current_bosses_df_an1['boss_fullname_with_tier'] = current_bosses_df_an1['boss_base_name'] + \
                                                               current_bosses_df_an1['boss_tier'].astype(str) + "阶"
            current_bosses_df_an1.loc[current_bosses_df_an1['boss_tier'] == 0, 'boss_fullname_with_tier'] = \
            current_bosses_df_an1['boss_base_name']
            if 'boss_fullname_with_tier' in current_bosses_df_an1.columns:
                avg_hero_star_per_report = current_heroes_df.groupby(get_col("reportId"))[
                    get_col("star_level")].mean().reset_index(name='avg_hero_star')
                merged_boss_hero_star = pd.merge(
                    current_bosses_df_an1[[get_col("reportId"), 'boss_fullname_with_tier']], avg_hero_star_per_report,
                    on=get_col("reportId"), how='left').dropna(subset=['avg_hero_star'])
                if not merged_boss_hero_star.empty:
                    avg_star_for_boss = merged_boss_hero_star.groupby('boss_fullname_with_tier')[
                        'avg_hero_star'].mean().round(2).reset_index(name='匹配神将的平均星级').sort_values(
                        by='匹配神将的平均星级', ascending=False)
                    if not avg_star_for_boss.empty:
                        fig_boss_match_star = px.bar(avg_star_for_boss, x='boss_fullname_with_tier',
                                                     y='匹配神将的平均星级',
                                                     title=f'各魔王匹配的神将平均星级{dynamic_title_suffix}',
                                                     labels={'boss_fullname_with_tier': '魔王名称 (含阶)'})
                        st.plotly_chart(fig_boss_match_star, use_container_width=True)
                else:
                    st.info(f"未能计算出魔王匹配的神将平均星级{dynamic_title_suffix}。")
            else:
                st.warning("未能成功处理魔王名称和阶数以进行分析一。")
        except Exception as e:
            st.error(f"分析一执行时出错: {e}")
    st.markdown("---")

    st.markdown("#### 2. 各星级神将最常匹配的魔王")
    if current_heroes_df.empty or current_bosses_df.empty:
        st.info("需要有效的神将和魔王数据才能进行分析二。")
    else:
        try:
            if not current_bosses_df_an1.empty and 'boss_fullname_with_tier' in current_bosses_df_an1.columns:  # Try to reuse from an1
                current_bosses_df_an2 = current_bosses_df_an1
            else:  # Reprocess if an1 failed or df is different
                boss_details_an2 = extract_boss_details(current_bosses_df[get_col("entity_name_final")])
                current_bosses_df_an2 = current_bosses_df.assign(boss_base_name=boss_details_an2['boss_base_name'],
                                                                 boss_tier=pd.to_numeric(boss_details_an2['boss_tier'],
                                                                                         errors='coerce').fillna(
                                                                     0).astype(int))
                current_bosses_df_an2['boss_fullname_with_tier'] = current_bosses_df_an2['boss_base_name'] + \
                                                                   current_bosses_df_an2['boss_tier'].astype(str) + "阶"
                current_bosses_df_an2.loc[current_bosses_df_an2['boss_tier'] == 0, 'boss_fullname_with_tier'] = \
                current_bosses_df_an2['boss_base_name']

            if 'boss_fullname_with_tier' in current_bosses_df_an2.columns:
                unique_bosses_per_report = current_bosses_df_an2[
                    [get_col("reportId"), 'boss_fullname_with_tier']].drop_duplicates(subset=[get_col("reportId")])
                merged_hero_boss_match = pd.merge(current_heroes_df[[get_col("reportId"), get_col("star_level")]],
                                                  unique_bosses_per_report, on=get_col("reportId"), how='inner')
                if not merged_hero_boss_match.empty:
                    hero_star_boss_counts = merged_hero_boss_match.groupby(
                        [get_col("star_level"), 'boss_fullname_with_tier']).size().reset_index(name='match_count')
                    idx = hero_star_boss_counts.groupby([get_col("star_level")])['match_count'].transform(max) == \
                          hero_star_boss_counts['match_count']
                    most_common_boss_for_hero_star = hero_star_boss_counts[idx].sort_values(
                        [get_col("star_level"), 'match_count'], ascending=[True, False]).drop_duplicates(
                        subset=[get_col("star_level")], keep='first')
                    total_matches_per_hero_star = merged_hero_boss_match.groupby(
                        get_col("star_level")).size().reset_index(name='total_star_matches')
                    result_df_analysis2 = pd.merge(most_common_boss_for_hero_star, total_matches_per_hero_star,
                                                   on=get_col("star_level"))
                    if not result_df_analysis2.empty:
                        result_df_analysis2['占比 (%)'] = (result_df_analysis2['match_count'] / result_df_analysis2[
                            'total_star_matches'] * 100).round(2)
                        result_df_analysis2 = result_df_analysis2.sort_values(by=get_col("star_level"))
                        result_df_analysis2.rename(
                            columns={'boss_fullname_with_tier': '最常匹配魔王', 'match_count': '匹配次数',
                                     'total_star_matches': '该星级神将总匹配数'}, inplace=True)
                        fig_hero_star_boss_match = px.bar(result_df_analysis2, x=get_col("star_level"), y='占比 (%)',
                                                          color='最常匹配魔王',
                                                          title=f'各星级神将最常匹配的魔王 (按占比){dynamic_title_suffix}',
                                                          labels={get_col("star_level"): "神将星级"},
                                                          text='最常匹配魔王')
                        fig_hero_star_boss_match.update_xaxes(type='category')
                        st.plotly_chart(fig_hero_star_boss_match, use_container_width=True)
                        # st.markdown(f"##### 各星级神将最常匹配魔王详情{dynamic_title_suffix}:") # Removed dataframe
                        # st.dataframe(result_df_analysis2[[get_col("star_level"), '最常匹配魔王', '占比 (%)']]) # Removed dataframe
                    else:
                        st.info(f"未能计算出各星级神将最常匹配的魔王{dynamic_title_suffix}。")
                else:
                    st.info(f"未能合并神将和魔王数据进行分析二{dynamic_title_suffix}。")
            else:
                st.warning("未能成功处理魔王名称和阶数以进行分析二。")
        except Exception as e:
            st.error(f"分析二执行时出错: {e}")
    st.markdown("---")

    st.markdown("#### 3. 特定星级神将的魔王匹配分布")
    hero_star_options_an3 = sorted(current_heroes_df[get_col("star_level")].dropna().unique().tolist())
    if not hero_star_options_an3:
        st.info("无神将星级数据可供选择。")
    else:
        selected_hero_star_an3 = st.selectbox("选择一个神将星级查看其匹配的魔王分布:", options=hero_star_options_an3,
                                              index=0, key="an3_hero_star_select")
        # Ensure current_bosses_df_an2 is defined and processed before this point if an1/an2 had issues
        if selected_hero_star_an3 is not None and not current_bosses_df.empty:
            # Try to use an2's processed df, or reprocess current_bosses_df if an2 is not ready
            processed_bosses_for_an3 = pd.DataFrame()
            if not current_bosses_df_an2.empty and 'boss_fullname_with_tier' in current_bosses_df_an2.columns:
                processed_bosses_for_an3 = current_bosses_df_an2
            elif not current_bosses_df.empty:  # Fallback to re-process current_bosses_df if an2 was empty
                try:
                    boss_details_an3_fallback = extract_boss_details(current_bosses_df[get_col("entity_name_final")])
                    processed_bosses_for_an3 = current_bosses_df.assign(
                        boss_base_name=boss_details_an3_fallback['boss_base_name'],
                        boss_tier=pd.to_numeric(boss_details_an3_fallback['boss_tier'], errors='coerce').fillna(
                            0).astype(int))
                    processed_bosses_for_an3['boss_fullname_with_tier'] = processed_bosses_for_an3['boss_base_name'] + \
                                                                          processed_bosses_for_an3['boss_tier'].astype(
                                                                              str) + "阶"
                    processed_bosses_for_an3.loc[
                        processed_bosses_for_an3['boss_tier'] == 0, 'boss_fullname_with_tier'] = \
                    processed_bosses_for_an3['boss_base_name']
                except:
                    pass  # If fallback also fails, it will remain empty

            if not processed_bosses_for_an3.empty and 'boss_fullname_with_tier' in processed_bosses_for_an3.columns:
                specific_star_heroes_df = current_heroes_df[
                    current_heroes_df[get_col("star_level")] == selected_hero_star_an3]
                if not specific_star_heroes_df.empty:
                    unique_bosses_per_report_an3 = processed_bosses_for_an3[
                        [get_col("reportId"), 'boss_fullname_with_tier']].drop_duplicates(subset=[get_col("reportId")])
                    merged_specific_hero_boss = pd.merge(specific_star_heroes_df[[get_col("reportId")]],
                                                         unique_bosses_per_report_an3, on=get_col("reportId"),
                                                         how='inner')
                    if not merged_specific_hero_boss.empty:
                        boss_counts_for_star = merged_specific_hero_boss[
                            'boss_fullname_with_tier'].value_counts().reset_index()
                        boss_counts_for_star.columns = ['魔王名称 (含阶)', '匹配次数']
                        total_matches_for_selected_star = boss_counts_for_star['匹配次数'].sum()
                        boss_counts_for_star['占比 (%)'] = (
                                    boss_counts_for_star['匹配次数'] / total_matches_for_selected_star * 100).round(2)
                        boss_counts_for_star = boss_counts_for_star.sort_values(by='占比 (%)', ascending=False)
                        st.write(f"**{selected_hero_star_an3}星神将** 匹配的魔王分布{dynamic_title_suffix}:")
                        fig_specific_star_boss_dist = px.pie(boss_counts_for_star, values='占比 (%)',
                                                             names='魔王名称 (含阶)',
                                                             title=f'{selected_hero_star_an3}星神将遇到的魔王占比{dynamic_title_suffix}',
                                                             hole=.3)
                        fig_specific_star_boss_dist.update_traces(textposition='inside', textinfo='percent+label')
                        st.plotly_chart(fig_specific_star_boss_dist, use_container_width=True)
                    else:
                        st.info(f"{selected_hero_star_an3}星神将没有匹配到任何魔王 (基于当前筛选)。")
                else:
                    st.info(f"沒有找到 {selected_hero_star_an3}星的神将数据 (基于当前筛选)。")
            else:
                st.warning("无法进行分析三，因为魔王数据处理不完整或当前无魔王数据。")
        elif selected_hero_star_an3 is None:
            st.info("请选择一个神将星级。")
# ... (后续的探索性分析 Tab 和 筛选后原始数据 Tab) ...

# with tabs[6]:  # 新的 "探索性分析" Tab 的索引是 6
#     st.subheader("自由探索性数据分析")
#
#     if filtered_df.empty:
#         st.info("请先通过侧边栏筛选数据，或确保有数据符合当前筛选条件。")
#         st.stop()  # 如果没有数据，探索也无从谈起
#
#     st.write("当前已筛选数据行数:", len(filtered_df))
#     st.dataframe(filtered_df.head())  # 显示筛选后数据的前几行作为参考
#
#     st.markdown("---")
#     st.write("#### 1. 查看列的描述性统计")
#
#     # 获取所有列名作为选项 (数值型和非数值型都可以用 describe)
#     all_columns = filtered_df.columns.tolist()
#     selected_column_for_describe = st.selectbox(
#         "选择一列查看描述性统计:",
#         options=all_columns,
#         key="desc_col_select"  # 给组件一个唯一的key
#     )
#     if selected_column_for_describe:
#         try:
#             st.write(f"**'{selected_column_for_describe}' 列的描述性统计:**")
#             st.dataframe(filtered_df[selected_column_for_describe].describe(include='all'))  # include='all' 对非数值型也有效
#         except Exception as e:
#             st.error(f"计算描述性统计时出错: {e}")
#
#     st.markdown("---")
#     st.write("#### 2. 简单双变量图表")
#
#     # 获取数值型列用于 x, y 轴和颜色/大小等
#     numeric_cols_for_plot = filtered_df.select_dtypes(include=['number']).columns.tolist()
#     # 获取类别型/对象型列用于 x 轴 (类别) 或颜色/分组等
#     categorical_cols_for_plot = filtered_df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
#
#     col1_plot, col2_plot, col3_plot = st.columns(3)
#     with col1_plot:
#         x_axis_col = st.selectbox("选择 X 轴 (通常是类别或数值):", options=['无'] + all_columns, key="x_axis")
#     with col2_plot:
#         y_axis_col = st.selectbox("选择 Y 轴 (通常是数值):", options=['无'] + numeric_cols_for_plot, key="y_axis")
#     with col3_plot:
#         chart_type = st.selectbox("选择图表类型:", ["条形图", "散点图", "箱线图", "直方图"], key="chart_type")
#
#     color_col = st.selectbox("选择颜色分组依据 (可选, 类别型列):",
#                              options=['无'] + categorical_cols_for_plot + numeric_cols_for_plot, key="color_axis")
#
#     # 只有当 X 轴和 Y 轴（对于需要Y轴的图表）都被选择时才绘图
#     can_plot = False
#     if chart_type in ["条形图", "散点图", "箱线图"] and x_axis_col != '无' and y_axis_col != '无':
#         can_plot = True
#     elif chart_type == "直方图" and x_axis_col != '无':  # 直方图只需要 X 轴
#         can_plot = True
#
#     if can_plot:
#         try:
#             fig_explore = None
#             title_explore = f"{chart_type}: '{x_axis_col}' vs '{y_axis_col if y_axis_col != '无' else ''}'"
#             if color_col != '无':
#                 title_explore += f" (按 '{color_col}' 分色)"
#
#             plot_kwargs = {'x': x_axis_col, 'title': title_explore}
#             if color_col != '无':
#                 plot_kwargs['color'] = color_col
#
#             if chart_type == "条形图":
#                 # 条形图通常 X 是类别, Y 是数值的聚合 (如均值、总和)
#                 # 这里我们简化为直接画，如果X是类别，Y是数值，Pandas绘图后端会自动处理
#                 # 或者我们需要用户选择聚合方式
#                 # 为简单起见，我们假设用户会选择合适的X和Y
#                 # 如果X是类别，Y是数值，Plotly Express 的 bar 会默认求和或计数（取决于y的数据类型）
#                 # 为了得到均值，需要预先 groupby
#                 # st.caption("提示: 条形图若X为类别，Y为数值，默认显示Y的总和或计数。如需均值等，需额外处理。")
#                 fig_explore = px.bar(filtered_df, y=y_axis_col, **plot_kwargs)
#             elif chart_type == "散点图":
#                 fig_explore = px.scatter(filtered_df, y=y_axis_col, **plot_kwargs)
#             elif chart_type == "箱线图":
#                 # 箱线图通常 X 是类别, Y 是数值
#                 fig_explore = px.box(filtered_df, y=y_axis_col, **plot_kwargs)
#             elif chart_type == "直方图":
#                 # 直方图只需要 X 轴 (数值型)
#                 if x_axis_col in numeric_cols_for_plot:
#                     plot_kwargs.pop('y', None)  # 移除 y (如果存在)
#                     fig_explore = px.histogram(filtered_df, **plot_kwargs)
#                 else:
#                     st.warning(f"直方图的 X 轴 ('{x_axis_col}') 应为数值型数据。")
#
#             if fig_explore:
#                 st.plotly_chart(fig_explore, use_container_width=True)
#         except Exception as e:
#             st.error(f"绘制图表时出错: {e}")
#     elif (x_axis_col != '无' or y_axis_col != '无'):  # 如果选了轴但不能画图
#         st.info("请为所选图表类型选择合适的 X 轴和 Y 轴。")

# ... (with tabs[5]: # 筛选后原始数据 Tab ... 及其后续代码) ...
st.write('---')
st.write('应用运行完毕。')
