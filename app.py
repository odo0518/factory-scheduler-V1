import streamlit as st

import pandas as pd

import numpy as np

from datetime import datetime, time

import io

import math

import re



# ==========================================

# 1. 全域配置

# ==========================================

SYSTEM_VERSION = "v23.1 (Fix: Multi-Gap Filling & Strict Setup Display)"



# 線外資源

OFFLINE_CONFIG_MAP = {

    "超音波": ("線外-超音波熔接", 1), 

    "LS": ("線外-組裝前LS", 2),

    "雷射": ("線外-組裝前LS", 2),

    "PT": ("線外-PT", 1),

    "PKM": ("線外-線邊組裝", 2),

    "裝配": ("線外-線邊組裝", 2),

    "組裝": ("線外-線邊組裝", 2),

    "AS": ("線外-線邊組裝", 2)

}

OFFLINE_DEFAULTS = list(OFFLINE_CONFIG_MAP.keys())



def get_base_model(product_id):

    if pd.isna(product_id): return ""

    s = str(product_id).strip().split('/')[0].strip()

    parts = s.split('-')

    if len(parts) >= 2 and parts[0].upper() == 'N':

        return f"{parts[0]}-{parts[1]}"

    return s



def parse_time_to_mins(time_str):

    try:

        t = datetime.strptime(time_str, "%H:%M")

        return t.hour * 60 + t.minute

    except: return 480 



def create_line_mask(start_str, end_str, days=14):

    total_minutes = days * 24 * 60

    mask = np.zeros(total_minutes, dtype=bool)

    start_min = parse_time_to_mins(start_str)

    end_min = parse_time_to_mins(end_str)

    breaks = [(600, 605), (720, 780), (900, 905), (1020, 1050)]

    for day in range(days):

        offset = day * 1440

        if end_min > start_min:

            mask[offset + start_min : offset + end_min] = True

            for b_s, b_e in breaks:

                mask[offset + b_s : offset + b_e] = False

    return mask



def format_time_str(minute_idx):

    d = (minute_idx // 1440) + 1

    m = minute_idx % 1440

    return f"D{d} {m//60:02d}:{m%60:02d}"



def extract_line_num(val):

    match = re.search(r'LINE(\d+)', str(val).upper().replace(' ', ''))

    return int(match.group(1)) if match else 0



def get_sequence(val):

    try:

        match = re.search(r'(\d+)', str(val))

        if match: return int(match.group(1))

        return 0 

    except: return 0



# ==========================================

# 2. 規則引擎

# ==========================================

class RuleEngine:

    def __init__(self, df_rules):

        self.rules = []          

        self.fixed_lines = set() 

        self.product_binding = {} 

        self.parse_rules(df_rules)



    def parse_rules(self, df):

        if df is None: return

        df.columns = df.columns.astype(str).str.replace(r'[\n\r\s]', '', regex=True)

        

        c_type = next((c for c in df.columns if '彈性' in c or '固定' in c), None)

        c_line = next((c for c in df.columns if '線別' in c and '彈性' not in c), None)

        c_prod = next((c for c in df.columns if '產品' in c), None)

        c_proc = next((c for c in df.columns if '領料' in c or '製程' in c), None)



        if not (c_type and c_line and c_prod): return



        for _, row in df.iterrows():

            l_type = str(row[c_type]).strip()

            l_name = str(row[c_line]).strip()

            p_pat = str(row[c_prod]).strip().replace('*', '') 

            proc = str(row[c_proc]).strip() if c_proc and not pd.isna(row[c_proc]) else ""

            

            l_idx = extract_line_num(l_name) - 4

            if l_idx < 0: continue



            if '固定' in l_type:

                self.fixed_lines.add(l_idx)

            

            if p_pat:

                if not proc or proc in ['工單發料', 'nan', '']:

                    self.product_binding[p_pat] = l_idx



            self.rules.append({

                'line_idx': l_idx, 'pattern': p_pat, 'process': proc, 'type': l_type

            })



    def get_assignment(self, product_id, process_type):

        for r in self.rules:

            if r['process'] and r['process'] not in ['工單發料', 'nan', '']:

                if r['pattern'] in str(product_id) and r['process'] in str(process_type):

                    return r['line_idx']

        return None



    def get_product_binding(self, product_id):

        for pat, l_idx in self.product_binding.items():

            if pat in str(product_id): return l_idx

        return None



    def can_line_accept_product(self, line_idx, product_id):

        # 1. 固定線潔癖 (只接白名單)

        if line_idx in self.fixed_lines:

            for r in self.rules:

                if r['line_idx'] == line_idx and r['pattern'] in str(product_id):

                    return True

            return False 

        

        # 2. 彈性線 (不搶固定線的單，除非規則允許)

        for pat, bound_line in self.product_binding.items():

            if pat in str(product_id):

                if line_idx != bound_line: return False

        return True 



# ==========================================

# 3. 資料讀取

# ==========================================

def load_and_clean_data(uploaded_file):

    try:

        xls = pd.read_excel(uploaded_file, sheet_name=None)

        df_ord = next((df for k,df in xls.items() if '工單' in df.columns or '產品' in str(df.columns)), None)

        df_rule = next((df for k,df in xls.items() if '線別' in str(df.columns) or '彈性' in str(df.columns)), None)

        

        if df_ord is None: return None, None, "缺少工單資料表"

        if df_rule is None: return None, None, "缺少規則資料表"

        

        engine = RuleEngine(df_rule)

        df = df_ord.copy()

        df.columns = df.columns.astype(str).str.replace(r'[\n\s]', '', regex=True)

        col_map = {}

        for c in df.columns:

            if '工單' in c: col_map[c] = 'Order_ID'

            elif '產品' in c: col_map[c] = 'Product_ID'

            elif '預定' in c: col_map[c] = 'Qty' 

            elif '人數' in c: col_map[c] = 'Manpower_Req' 

            elif '工時' in c: col_map[c] = 'Total_Man_Minutes' 

            elif '項次' in c: col_map[c] = 'Priority'

            elif '領料' in c: col_map[c] = 'Process_Type'

            elif '備註' in c: col_map[c] = 'Remarks'

            elif '急單' in c: col_map[c] = 'Rush_Col'

            elif '指定' in c: col_map[c] = 'Line_Col'

        df = df.rename(columns=col_map)

        

        for c in ['Qty', 'Manpower_Req', 'Total_Man_Minutes']:

            if c in df.columns:

                df[c] = pd.to_numeric(df[c].astype(str).str.replace(',', ''), errors='coerce').fillna(0)

            else: df[c] = 0

            

        df = df[df['Qty'] > 0]

        df['Base_Model'] = df['Product_ID'].apply(get_base_model)

        

        def classify_row(row):

            prod = str(row['Product_ID'])

            proc = str(row['Process_Type'])

            assigned_line = engine.get_assignment(prod, proc)

            if assigned_line is not None: return False, assigned_line + 4, "Online", 0

            

            is_offline_kw = False

            offline_info = ("", 0)

            for kw, (gname, limit) in OFFLINE_CONFIG_MAP.items():

                if kw in proc:

                    is_offline_kw = True

                    offline_info = (gname, limit)

                    break

            if is_offline_kw: return True, 0, offline_info[0], offline_info[1]



            orig_target = extract_line_num(row.get('Line_Col', ''))

            if orig_target == 0: orig_target = extract_line_num(row.get('Remarks', ''))

            if orig_target == 0:

                bound_line = engine.get_product_binding(prod)

                if bound_line is not None: orig_target = bound_line + 4

            

            return False, orig_target, "Online", 0



        temp = df.apply(classify_row, axis=1)

        df['Is_Offline'] = temp.apply(lambda x: x[0])

        df['Target_Line'] = temp.apply(lambda x: x[1])

        df['Process_Category'] = temp.apply(lambda x: x[2])

        df['Concurrency_Limit'] = temp.apply(lambda x: x[3])



        if 'Rush_Col' not in df.columns: df['Rush_Col'] = ''

        df['Is_Rush'] = df['Rush_Col'].astype(str).str.contains('急單', na=False) | df['Remarks'].astype(str).str.contains('急單', na=False)

        df['Sequence'] = df['Remarks'].apply(get_sequence)



        return df, engine, None

    except Exception as e: return None, None, str(e)



# ==========================================

# 4. 報表計算

# ==========================================

def analyze_idle_manpower(timeline, masks, total_mp, max_min):

    global_mask = np.zeros(max_min, dtype=bool)

    for m in masks:

        l = min(len(m), max_min)

        global_mask[:l] |= m[:l]

    records = []

    curr_excess, start_t = -1, -1

    for t in range(max_min):

        if global_mask[t]:

            used = timeline[t]

            excess = total_mp - used

            if excess != curr_excess:

                if curr_excess > 0 and start_t != -1:

                    records.append({'開始時間': format_time_str(start_t), '結束時間': format_time_str(t), '持續分鐘': t-start_t, '閒置(多餘)人力': curr_excess})

                curr_excess, start_t = excess, t

        else:

            if curr_excess > 0 and start_t != -1:

                records.append({'開始時間': format_time_str(start_t), '結束時間': format_time_str(t), '持續分鐘': t-start_t, '閒置(多餘)人力': curr_excess})

            curr_excess, start_t = -1, -1

    return pd.DataFrame(records)



def calculate_daily_efficiency(timeline, masks, total_mp, results, days):

    recs = []

    for d in range(days):

        s, e = d*1440, (d+1)*1440

        std = np.sum(masks[0][s:e])

        used = np.sum(timeline[s:e])

        cap = total_mp * std

        eff = (used/cap*100) if cap > 0 else 0

        qty = sum(r['數量'] for r in results if r['狀態']=='OK' and s <= r['排序用'] < e)

        sug_mp = math.ceil(used / (std * 0.95)) if std > 0 else 0

        diff = sug_mp - total_mp

        sug_str = f"增 {diff}" if diff > 0 else f"減 {abs(diff)}"

        recs.append({'日期': f'D{d+1}', '當日標準工時(分)': std, '現有人力': total_mp, '建議人力(95%效)': sug_mp, '調度建議': sug_str, '實際產出人時': used, '總產出數': int(qty), '全廠效率(%)': round(eff, 2)})

    return pd.DataFrame(recs)



def calculate_line_utilization(matrix, masks, lines, days):

    recs = []

    for d in range(days):

        s, e = d*1440, (d+1)*1440

        row = {'日期': f'D{d+1}'}

        for i in range(lines):

            avail = np.sum(masks[i][s:e])

            busy = np.sum(matrix[i][s:e] & masks[i][s:e])

            row[f'Line {i+4} (%)'] = round(busy/avail*100, 1) if avail > 0 else 0

        recs.append(row)

    return pd.DataFrame(recs)



# ==========================================

# 5. 排程運算區 (Multi-Task Gap Filling + Strict Setup)

# ==========================================

def run_scheduler(df, engine, total_manpower, total_lines, std_changeover, similar_changeover, line_settings, offline_settings):

    MAX_MINUTES = 14 * 24 * 60 

    

    line_masks = []

    line_cumsums = []

    for setting in line_settings:

        m = create_line_mask(setting["start"], setting["end"], 14)

        line_masks.append(m)

        line_cumsums.append(np.cumsum(m))

    line_free_time = [parse_time_to_mins(setting["start"]) for setting in line_settings]

    line_last_model = {i: None for i in range(total_lines)}

    

    offline_mask = create_line_mask(offline_settings["start"], offline_settings["end"], 14)

    offline_cumsum = np.cumsum(offline_mask)

    offline_resource_usage = {} 

    

    timeline_manpower = np.zeros(MAX_MINUTES, dtype=int)

    line_usage_matrix = np.zeros((total_lines, MAX_MINUTES), dtype=bool)

    

    order_finish_times = {} 

    results = []



    rush_ids = df[df['Is_Rush']]['Order_ID'].unique()

    df['Order_Is_Rush'] = df['Order_ID'].isin(rush_ids)

    

    # 建立 ID 與分池

    all_tasks = df.to_dict('records')

    for i, t in enumerate(all_tasks): t['Pool_ID'] = i

    

    pool_rush = [t for t in all_tasks if t['Order_Is_Rush']]

    pool_fixed = []

    pool_normal = []

    

    for t in all_tasks:

        if t['Order_Is_Rush']: continue

        

        is_fixed = False

        if t['Target_Line'] > 0 and (t['Target_Line']-4) in engine.fixed_lines:

            is_fixed = True

        elif t['Target_Line'] == 0:

            bound = engine.get_product_binding(t['Base_Model'])

            if bound is not None and bound in engine.fixed_lines:

                is_fixed = True

        

        if is_fixed: pool_fixed.append(t)

        else: pool_normal.append(t)



    # 排序

    pool_rush.sort(key=lambda x: (x['Sequence'], x['Priority']))

    pool_fixed.sort(key=lambda x: (x['Target_Line'], x['Sequence'], x['Priority'])) 

    pool_normal.sort(key=lambda x: (x['Sequence'], x['Priority']))



    # ---------------- 核心函數 ----------------

    def check_line_permission(l_idx, base_model, has_target_line, target_line_val):

        if has_target_line:

            return l_idx == (target_line_val - 4)

        return engine.can_line_accept_product(l_idx, base_model)



    def get_setup(l_idx, model, start_time):

        if line_last_model[l_idx] is None: return 0

        curr_day = start_time // 1440

        prev_finish = line_free_time[l_idx] 

        if (curr_day > (prev_finish // 1440)): return 0 

        return similar_changeover if line_last_model[l_idx] == model else std_changeover



    def find_earliest_slot(task, l_idx, min_start_time):

        manpower = int(task['Manpower_Req'])

        prod_duration = int(np.ceil(float(task['Total_Man_Minutes']) / manpower)) if manpower > 0 else 0

        

        if task['Is_Offline']:

            t_search = min_start_time

            mask = offline_mask

            cumsum = offline_cumsum

            res_group = task['Process_Category']

            res_limit = task['Concurrency_Limit']

            if res_group not in offline_resource_usage: 

                offline_resource_usage[res_group] = np.zeros(MAX_MINUTES, dtype=int)

        else:

            t_search = max(line_free_time[l_idx], min_start_time)

            mask = line_masks[l_idx]

            cumsum = line_cumsums[l_idx]



        while t_search < MAX_MINUTES - prod_duration: 

            if not mask[t_search]:

                t_search += 1

                continue

            

            this_setup = 0

            if not task['Is_Offline']:

                this_setup = get_setup(l_idx, task['Base_Model'], t_search)

            

            total_need = this_setup + prod_duration

            s_val = cumsum[t_search]

            t_val = s_val + total_need

            if t_val > cumsum[-1]: return None

            t_end = np.searchsorted(cumsum, t_val)

            

            if np.any(mask[t_search:t_end]):

                valid_slice = slice(t_search, t_end)

                time_mask = mask[valid_slice]

                curr_mp = timeline_manpower[valid_slice][time_mask]

                max_mp = np.max(curr_mp) if len(curr_mp) > 0 else 0

                

                res_ok = True

                if task['Is_Offline']:

                    curr_res = offline_resource_usage[res_group][valid_slice][time_mask]

                    max_res = np.max(curr_res) if len(curr_res) > 0 else 0

                    if max_res >= res_limit: res_ok = False

                

                if res_ok and (max_mp + manpower <= total_manpower):

                    return t_search, t_end, this_setup, None

                else:

                    t_search += 5 

            else:

                t_search += 5 

        return None



    def book_slot(task, l_idx, slot_info, log_msg=""):

        start, end, setup, _ = slot_info

        manpower = int(task['Manpower_Req'])

        

        # ★★★ 修正 2: 換線時間顯示 ★★★

        # 排程顯示的開始時間 = 實際上工時間 = Start + Setup

        # 但系統預留時間是從 Start 開始到 End

        # 為了報表正確，預計開始時間應該顯示 Start，並註明包含換線

        # 或者：預計開始 = Start (包含換線)，完工時間 = End

        # 使用者要求：完工時間 11:46, 換線 10 分 -> 下一張預計開始 11:56

        # 所以這裡的 Start 已經是 "包含了換線時間" 的起點嗎？

        # find_earliest_slot 回傳的 `start` 是 "能開始塞入(含換線)的時間點"

        # 所以顯示上：預計開始 = start + setup

        

        display_start = start + setup

        

        if task['Is_Offline']:

            mask_slice = offline_mask[start:end]

            timeline_manpower[start:end][mask_slice] += manpower

            res_group = task['Process_Category']

            offline_resource_usage[res_group][start:end][mask_slice] += 1

            display_line = res_group

        else:

            mask_slice = line_masks[l_idx][start:end]

            timeline_manpower[start:end][mask_slice] += manpower

            line_usage_matrix[l_idx, start:end] = True

            line_free_time[l_idx] = end

            line_last_model[l_idx] = task['Base_Model']

            display_line = f"Line {l_idx+4}"

            

        order_finish_times[(str(task['Order_ID']), task['Sequence'])] = end

        

        results.append({

            '產線': display_line,

            '工單': task['Order_ID'], '產品': task['Product_ID'], 

            '數量': task['Qty'], '類別': '線外' if task['Is_Offline'] else '流水線', 

            '換線(分)': setup, '需求人力': manpower, 

            '預計開始': format_time_str(display_start), # 顯示實際生產開始時間

            '完工時間': format_time_str(end), 

            '線佔用(分)': (end - start), '狀態': 'OK', '排序用': end,

            '備註': task.get('Remarks', ''), '指定線': task.get('Line_Col', ''),

            '急單': 'Yes' if task.get('Order_Is_Rush') else '', '判斷': log_msg

        })



    def check_dependency(task):

        if task['Sequence'] <= 1: return True, parse_time_to_mins(line_settings[0]["start"])

        prev_key = (str(task['Order_ID']), task['Sequence'] - 1)

        if prev_key in order_finish_times:

            return True, order_finish_times[prev_key]

        return False, 0



    # ==================================================

    # STEP 1 & 2: 急單優先 + 多工單填補 (Multi-Gap Fill)

    # ==================================================

    while pool_rush:

        task = pool_rush.pop(0)

        is_ready, dep_time = check_dependency(task)

        

        if not is_ready:

            pool_rush.append(task)

            # 安全機制... (省略，與 v23 相同)

            if len(pool_rush) > 0 and all(not check_dependency(t)[0] for t in pool_rush):

                pass

            continue



        if task['Is_Offline']:

            min_start = max(dep_time, parse_time_to_mins(offline_settings["start"]))

            slot = find_earliest_slot(task, -1, min_start)

            if slot: book_slot(task, -1, slot, "Rush_Offline")

        else:

            t_req = task['Target_Line']

            candidates = [t_req-4] if t_req > 0 else [l for l in range(total_lines) if check_line_permission(l, task['Base_Model'], False, 0)]

            

            best_opt = None

            for l_idx in candidates:

                # ★★★ 修正 1: 多工單填補迴圈 (Multi-Fill) ★★★

                # 只要空隙夠大，就一直填，填到不能填為止

                while True:

                    gap = dep_time - line_free_time[l_idx]

                    if gap <= 30: break # 空隙太小，不填了，直接排急單

                    

                    # 尋找最佳填補者 (Normal Pool)

                    best_filler = None # (idx, task, slot)

                    

                    for n_idx, n_task in enumerate(pool_normal):

                        if n_task['Is_Offline']: continue

                        if not check_line_permission(l_idx, n_task['Base_Model'], False, 0): continue

                        n_ready, n_dep = check_dependency(n_task)

                        if not n_ready: continue

                        

                        # 試算

                        n_start = max(line_free_time[l_idx], n_dep)

                        n_slot = find_earliest_slot(n_task, l_idx, n_start)

                        

                        if n_slot:

                            n_end = n_slot[1]

                            # 允許稍微延後 (10%)

                            if n_end <= dep_time + (gap * 0.1):

                                # 貪婪：找時間最接近 gap (塞最滿) 的單

                                # 或者找最早能開始的單? 這裡選 "塞得最剛好" -> 減少剩餘空隙

                                # 但為了簡單且高效，我們選 "第一張能塞進去的" (First Fit) 

                                # 或者 "耗時最長但小於 gap" 的 (Best Fit)

                                f_dur = n_end - n_start

                                if best_filler is None or f_dur > (best_filler[2][1] - best_filler[2][0]):

                                    best_filler = (n_idx, n_task, n_slot)

                    

                    if best_filler:

                        f_idx, f_task, f_slot = best_filler

                        book_slot(f_task, l_idx, f_slot, "Rush_Multi_Fill")

                        pool_normal.pop(f_idx)

                        # 填補後 line_free_time 推進了，迴圈繼續檢查剩餘 gap

                    else:

                        break # 找不到能填的單了，跳出填補迴圈



                # 正常排急單 (此時 gap 應該已經被填到最小)

                my_start = max(dep_time, line_free_time[l_idx])

                slot = find_earliest_slot(task, l_idx, my_start)

                if slot:

                    if best_opt is None or slot[1] < best_opt[0]:

                        best_opt = (slot[1], l_idx, slot)

            

            if best_opt:

                book_slot(task, best_opt[1], best_opt[2], "Rush_On")

            else:

                results.append({'工單': task['Order_ID'], '狀態': 'Fail', '備註': '急單資源不足'})



    # ==================================================

    # STEP 3: 固定線塞滿 (Fixed Line Saturation)

    # ==================================================

    fixed_tasks_map = {} 

    for t in pool_fixed:

        target = -1

        if t['Target_Line'] > 0: target = t['Target_Line'] - 4

        else:

            b = engine.get_product_binding(t['Base_Model'])

            if b is not None: target = b

        

        if target != -1:

            if target not in fixed_tasks_map: fixed_tasks_map[target] = []

            fixed_tasks_map[target].append(t)

            

    for l_idx, tasks in fixed_tasks_map.items():

        tasks.sort(key=lambda x: x['Sequence'])

        for task in tasks:

            is_ready, dep_time = check_dependency(task)

            min_start = max(dep_time, line_free_time[l_idx])

            slot = find_earliest_slot(task, l_idx, min_start)

            if slot: book_slot(task, l_idx, slot, "Fixed_Saturation")

            else: results.append({'工單': task['Order_ID'], '狀態': 'Fail', '備註': '固定線資源不足'})



    # ==================================================

    # STEP 4: 一般工單貪婪填充 (Normal Greedy)

    # ==================================================

    while pool_normal:

        lines_status = sorted(range(total_lines), key=lambda x: line_free_time[x])

        global_best = None 

        

        for l_idx in lines_status:

            line_ready_time = line_free_time[l_idx]

            if global_best and line_ready_time > global_best[1] + 1440: continue



            for t_idx, task in enumerate(pool_normal):

                if task['Is_Offline']: continue 

                if not check_line_permission(l_idx, task['Base_Model'], task['Target_Line']>0, task['Target_Line']): continue

                

                is_ready, dep_time = check_dependency(task)

                if not is_ready: continue

                

                start_time = max(line_ready_time, dep_time)

                gap = start_time - line_ready_time

                if gap > 2880: continue 

                

                slot = find_earliest_slot(task, l_idx, start_time)

                if slot:

                    finish = slot[1]

                    setup = slot[2]

                    score = (gap * 100) + finish + (setup * 5)

                    if global_best is None or score < global_best[0]:

                        global_best = (score, finish, l_idx, t_idx, slot)

            

            if global_best and (global_best[4][0] - line_ready_time == 0) and global_best[4][2] == 0:

                break



        # 線外

        for t_idx, task in enumerate(pool_normal):

            if not task['Is_Offline']: continue

            is_ready, dep_time = check_dependency(task)

            if not is_ready: continue

            

            min_start = max(dep_time, parse_time_to_mins(offline_settings["start"]))

            slot = find_earliest_slot(task, -1, min_start)

            if slot:

                finish = slot[1]

                if global_best is None or slot[0] < global_best[0]: 

                    global_best = (slot[0], finish, -1, t_idx, slot)



        if global_best:

            _, _, l_idx, t_idx, slot = global_best

            book_slot(pool_normal[t_idx], l_idx, slot, "Normal_Greedy")

            pool_normal.pop(t_idx)

        else:

            if all(not check_dependency(t)[0] for t in pool_normal):

                for t in pool_normal: results.append({'工單': t['Order_ID'], '狀態': 'Fail', '備註': '死鎖'})

                break

            if pool_normal:

                for t in pool_normal: results.append({'工單': t['Order_ID'], '狀態': 'Fail', '備註': '資源不足'})

                break



    if results:

        last = max([r['排序用'] for r in results if r.get('狀態')=='OK'], default=0)

        days = (last // 1440) + 1

        df_res = pd.DataFrame(results)

        df_eff = calculate_daily_efficiency(timeline_manpower, line_masks, total_manpower, results, days)

        df_util = calculate_line_utilization(line_usage_matrix, line_masks, total_lines, days)

        df_idle = analyze_idle_manpower(timeline_manpower, line_masks, total_manpower, days*1440)

        return df_res, df_idle, df_eff, df_util

        

    return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()



# ==========================================

# 6. UI

# ==========================================

st.set_page_config(page_title="AI 智能排程系統", layout="wide")

st.title(f"🏭 {SYSTEM_VERSION}")



with st.sidebar:

    st.header("⚙️ 全域參數")

    total_manpower = st.number_input("全廠總人力", value=50)

    total_lines = st.number_input("產線數量", value=5)

    c1, c2 = st.columns(2)

    std_changeover = c1.number_input("標準換線", value=10)

    sim_changeover = c2.number_input("相似換線", value=5)

    

    line_settings = []

    with st.expander("產線時間", expanded=True):

        for i in range(total_lines):

            c1, c2 = st.columns(2)

            s = c1.time_input(f"L{i+4}起", time(8,0), key=f"s{i}")

            e = c2.time_input(f"L{i+4}迄", time(17,0), key=f"e{i}")

            line_settings.append({"start": s.strftime("%H:%M"), "end": e.strftime("%H:%M")})

    

    c1, c2 = st.columns(2)

    os = c1.time_input("線外起", time(8,0))

    oe = c2.time_input("線外迄", time(17,0))

    offline_settings = {"start": os.strftime("%H:%M"), "end": oe.strftime("%H:%M")}



f = st.file_uploader("上傳 Excel (含工單與規則)", type=['xlsx'])

if f:

    df, engine, err = load_and_clean_data(f)

    if err: st.error(err)

    else:

        with st.expander("規則檢視"):

            st.write("Fixed:", engine.fixed_lines)

            st.write("Product Binding:", engine.product_binding)

            

        if st.button("Run"):

            res, idle, eff, util = run_scheduler(df, engine, total_manpower, total_lines, std_changeover, sim_changeover, line_settings, offline_settings)

            

            t1, t2, t3, t4 = st.tabs(["排程表", "效率", "稼動", "閒置"])

            with t1: st.dataframe(res, use_container_width=True)

            with t2: st.dataframe(eff, use_container_width=True)

            with t3: st.dataframe(util, use_container_width=True)

            with t4: st.dataframe(idle, use_container_width=True)



            out = io.BytesIO()

            with pd.ExcelWriter(out, engine='xlsxwriter') as writer:

                res.to_excel(writer, sheet_name="排程", index=False)

                eff.to_excel(writer, sheet_name="效率", index=False)

                util.to_excel(writer, sheet_name="稼動", index=False)

                idle.to_excel(writer, sheet_name="閒置", index=False)

            out.seek(0)

            st.download_button("下載報表", out, "Schedule_v23.1.xlsx")
