"""
Created on 2024/6/13 13:00
@author: TianMC
"""
import pandas as pd
import numpy as np
from datetime import datetime
from WindPy import w
import time
from dateutil.relativedelta import relativedelta
w.start()
w.isconnected()
import warnings
import os
warnings.filterwarnings('ignore')

# bench_code = '000300.SH'  # 000300.SH / 000905.SH
# fund_code = '000172.OF'   # 000311.OF / 000172.OF / 260112.OF
# start_date = '2020-01-01'
# end_date = '2024-06-10'  #
#
# used_data_path = r'\data\used_data'
# result_file_path = r'\data\ret_result'
# fund_weight_data_path = r'\data\fund_stock_weight'
# bench_weight_data_path = r'\data\bench_stock_weight'
# daily_ret_data_path = r'\data\fund_bench_daily_ret'
# semiyear_ret_data_path = r'\data\fund_bench_semiyear_ret'
#
#
# current_path = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在的路径
# used_data_path = current_path + used_data_path
# result_file_path = current_path + result_file_path
# fund_weight_data_path = current_path + fund_weight_data_path
# bench_weight_data_path = current_path + bench_weight_data_path
# daily_ret_data_path = current_path + daily_ret_data_path
# semiyear_ret_data_path = current_path + semiyear_ret_data_path


# 获取一段时间内的半年日期，xx0630/xx1231
def date_semi(start_dt, end_dt):
    # 半年日期
    start_year = int(start_dt[:4])
    end_year = int(end_dt[:4])
    dt_semi = pd.to_datetime([
        f"{year}-{month:02d}-{day:02d}"
        for year in range(start_year - 1, end_year + 1)
        for month, day in [(6, 30), (12, 31)]
    ])
    dt_semi = dt_semi[dt_semi <= datetime.strptime(end_dt, '%Y-%m-%d')]
    return dt_semi


# 数据提取框架，index为半年期交易日，col为全A市场+已摘牌股票(按照end_date日期提取的股票代码清单)
def data_structure(start_dt, end_dt, file_path, update_stockpool=False):
    """
    :param end_dt: 格式无所谓但是前四位必须为年份
    :param file_path:  数据提取框架存储位置
    :return: index为date，返回的股票池是end_dt对应的股票池，日期为每年的06-30/12-31（只有这个日期才有基金的半年报/年报）
    """
    # 半年日期
    dt_semi = date_semi(start_dt, end_dt)
    dt_semi = dt_semi.append(pd.DatetimeIndex([pd.to_datetime(end_dt)]))

    try:
        df = pd.read_csv(file_path + rf'\data_structure.csv', parse_dates=True, index_col=0)
        print(df, dt_semi)
        print('正在从本地获取数据框架文件：')

        # 更新数据提取框架
        if df.index[-1] != dt_semi[-1] and not update_stockpool:
            print('正在更新数据提取框架日期:')

            df = pd.DataFrame(index=dt_semi, columns=df.columns)
            df.index.name = 'date'
            df.to_csv(file_path + rf'\data_structure.csv')
            print('数据提取框架日期更新完毕！')
        if update_stockpool:
            print('正在更新数据提取框架日期和股票池：')
            # 全A市场股票
            stcodes_a = w.wset("sectorconstituent", f"date={end_dt};sectorid=a001010100000000;field=wind_code")

            # 已退市股票
            stcodes_delist = w.wset("sectorconstituent", f"date={end_dt};sectorid=a001010m00000000;field=wind_code")

            stcodes = stcodes_a.Data[0] + stcodes_delist.Data[0]
            df = pd.DataFrame(index=dt_semi, columns=stcodes)
            df.index.name = 'date'
            df.to_csv(file_path + rf'\data_structure.csv')
            print('数据提取框架日期和股票池已更新并保存本地！')

        return df

    except FileNotFoundError:
        print('正在从wind获取数据框架数据：')
        # 全A市场股票
        stcodes_a = w.wset("sectorconstituent", f"date={end_dt};sectorid=a001010100000000;field=wind_code")

        # 已退市股票
        stcodes_delist = w.wset("sectorconstituent", f"date={end_dt};sectorid=a001010m00000000;field=wind_code")

        stcodes = stcodes_a.Data[0] + stcodes_delist.Data[0]
        df = pd.DataFrame(index=dt_semi, columns=stcodes)
        df.index.name = 'date'
        df.to_csv(file_path + rf'\data_structure.csv')
        print('数据框架已保存本地！')
        return df
# struc = data_structure(start_date, end_date, file_path=used_data_path)
# print(struc)


# 获取数据提取框架的中证一级行业, index+1天的行业
def get_stock_ind(start_dt, end_dt, struc_path, ind_path):
    try:
        df_ori = pd.read_csv(ind_path + rf'\stock_ind.csv', encoding='utf-8', parse_dates=True, index_col=0)
        print('正在从本地获取行业文件：')

        # 半年日期
        dt_semi = date_semi(start_dt, end_dt)

        # 若原文件的最后一个日期不是对应的end_dt对应的最后一期
        if df_ori.index[-1] < dt_semi[-1]:
            print('正在更新本地数据:')

            # 只更新多余的日期对应的数据
            stocks_list = df_ori.columns.tolist()
            dt_semi_more = [dt for dt in dt_semi if dt not in df_ori.index]
            df = pd.DataFrame(index=dt_semi_more, columns=stocks_list)

            # 更新行业数据
            for dt in df.index:
                df.loc[dt] = w.wss(stocks_list, "industry_citic", f"tradeDate={dt + relativedelta(days=1)};industryType=1").Data[0]
            df_ori = pd.concat([df_ori, df])
            df_ori.dropna(how='all', inplace=True)  # 去掉没有数据的行
            df_ori.to_csv(ind_path + rf'\stock_ind.csv', encoding='utf-8')
            print('数据更新完毕！')
        return df_ori

    except FileNotFoundError:
        print('正在从wind获取行业数据：')
        df_structure = data_structure(start_dt=start_dt, end_dt=end_dt, file_path=struc_path)
        stocks_list = df_structure.columns.tolist()
        df = pd.DataFrame(index=df_structure.index, columns=stocks_list)
        for dt in df.index:
            df.loc[dt] = w.wss(stocks_list, "industry_citic", f"tradeDate={dt + relativedelta(days=1)};industryType=1").Data[0]
        df.dropna(how='all', inplace=True)
        df.to_csv(ind_path + rf'\stock_ind.csv', encoding='utf-8')
        print('行业数据已保存本地！')
        return df
# stocks_ind = get_stock_ind(start_dt=start_date, end_dt=end_date, struc_path=used_data_path, ind_path=used_data_path)
# print(stocks_ind)


# 获取基金/基准持仓情况,index为date, col=['stcode', 'weight', 'indname']
def stock_weight(start_dt, end_dt, wcode, stocks_ind, file_path, bench=False, fund=False):
    """
    :param wcode: wind bench代码,只能是单个code的字符
    :param dt: 每个调仓日期，返回该日期的基准的持股权重
    :return: index为date, columns=['stcode', 'weight', 'indname']3列的df,date列是datetime,每个dt的权重之和接近1
    """
    # 半年日期
    dt_semi = date_semi(start_dt, end_dt)

    try:
        # 若本地有该文件，则直接读取
        df = pd.read_csv(file_path + rf'\{wcode}.csv', parse_dates=True, index_col=0)
        print(f'正在从本地获取{wcode}持仓文件：')

        # 更新本地文件数据
        if df.index[-1] < dt_semi[-1]:
            print(f'正在更新{wcode}的持股成分权重:')
            st = time.time()
            dt_semi_more = [dt for dt in dt_semi if dt not in df.index]
            for dt in dt_semi_more:
                # 获取持仓股票的'stcode', 'weight', weight之和不超过1
                if bench:
                    tmp = w.wset("indexconstituent", f"date={dt};windcode={wcode};field=wind_code,i_weight")
                elif fund:
                    tmp = w.wset("allfundhelddetail", f"rptdate={dt};windcode={wcode};field=stock_code,proportiontonetvalue")
                else:
                    print('请输入bench=True 或者 fund=True')
                    break
                tmp1 = list(zip(*tmp.Data))
                df1 = pd.DataFrame(tmp1, columns=['stcode', 'weight'])
                df1['weight'] = df1['weight'] / 100
                df1['date'] = dt

                # 忽略基金持仓低于千分之一的股票，忽略BJ,HK股票
                if fund:
                    df1 = df1[df1['weight'] >= 0.001]
                    df1 = df1[~df1['stcode'].str.endswith(('BJ', 'HK'))]

                # 获取股票对应中证一级行业
                ind = stocks_ind.loc[dt].loc[df1['stcode'].tolist()].tolist()
                df1['indname'] = ind

                if len(df1) != 0:
                    df = pd.concat([df, df1])

            df.set_index('date', inplace=True)
            df.to_csv(file_path + rf'\{wcode}.csv', encoding='utf-8')
            print(f'更新{wcode}的持股成分权重完成，共耗时{time.time() - st}')

    except FileNotFoundError:
        print(f'正在获取{wcode}的持股成分权重:')
        st = time.time()
        df = pd.DataFrame()
        for dt in dt_semi:
            # 获取持仓股票的'stcode', 'weight', weight之和不超过1
            if bench:
                tmp = w.wset("indexconstituent", f"date={dt};windcode={wcode};field=wind_code,i_weight")
            elif fund:
                tmp = w.wset("allfundhelddetail", f"rptdate={dt};windcode={wcode};field=stock_code,proportiontonetvalue")
            else:
                print('请输入bench=True 或者 fund=True')
                break
            tmp1 = list(zip(*tmp.Data))
            df1 = pd.DataFrame(tmp1, columns=['stcode', 'weight'])
            df1['weight'] = df1['weight'] / 100
            df1['date'] = dt

            # 忽略基金持仓低于千分之一的股票，忽略BJ,HK股票
            if fund:
                df1 = df1[df1['weight'] >= 0.001]
                df1 = df1[~df1['stcode'].str.endswith(('BJ', 'HK'))]

            # 获取股票对应中证一级行业
            ind = stocks_ind.loc[dt].loc[df1['stcode'].tolist()].tolist()
            df1['indname'] = ind

            if len(df1) != 0:
                df = pd.concat([df, df1])
        df.set_index('date', inplace=True)
        df.to_csv(file_path + rf'\{wcode}.csv', encoding='utf-8')
        print(f'成功获取{wcode}的持股成分权重，共耗时{time.time() - st}')
    return df
# fund = stock_weight(start_dt=start_date, end_dt=end_date, wcode=fund_code, stocks_ind=stocks_ind, file_path=fund_weight_data_path, fund=True)
# print(fund)
# bench = stock_weight(start_dt=start_date, end_dt=end_date, wcode=bench_code, stocks_ind=stocks_ind, file_path=bench_weight_data_path, bench=True)
# print(bench)


# 获取股票池半年期的收益率, index +1 天到下一个index的区间内, 并不会更新股票池， index为date
def get_stock_ret(start_dt, end_dt, struc_path, ret_path):
    try:
        df = pd.read_csv(ret_path + rf'\stock_ret.csv', parse_dates=True, index_col=0, encoding='utf-8')
        print('正在从本地获取股票半年期收益文件：')

        # 获取end_dt和之前一年的半年期日期, 加上end_dt
        dt_semi = date_semi(start_dt, end_dt)
        dt_semi = dt_semi.append(pd.DatetimeIndex([pd.to_datetime(end_dt)]))

        if df.index[-1] < dt_semi[-1]:
            print('正在更新股票半年期收益数据:')
            dt_semi_more = [dt for dt in dt_semi if dt not in df.index]
            stcode_list = df.columns.tolist()
            df1 = pd.DataFrame(index=dt_semi_more, columns=stcode_list)

            # 更新行业数据
            for i, dt in enumerate(dt_semi_more):
                if dt != dt_semi_more[-1]:
                    dt_next = dt_semi_more[i + 1]
                    df1.loc[dt] = w.wss(stcode_list, "pct_chg_per", f"startDate={dt + relativedelta(days=1)};endDate={dt_next}").Data[0]
            df = pd.concat([df, df1])
            df = df.dropna(how='all')  # 删掉没有数据的行
            df.to_csv(ret_path + rf'\stock_ret.csv', encoding='utf-8')
            print('股票半年期收益数据更新完毕！')
        return df

    except FileNotFoundError:
        print('正在从wind获取股票半年期收益数据：')
        df_structure = data_structure(start_dt, end_dt, file_path=struc_path)
        dt_list = df_structure.index
        stcode_list = df_structure.columns.tolist()
        df = pd.DataFrame(index=dt_list, columns=stcode_list)

        for i, dt in enumerate(dt_list):
            if dt != dt_list[-1]:
                dt_next = dt_list[i+1]
                df.loc[dt] = w.wss(stcode_list, "pct_chg_per",f"startDate={dt + relativedelta(days=1)};endDate={dt_next}").Data[0]
        df.index.name = 'date'
        df = df.dropna(how='all')  # 删掉没有数据的行
        df.to_csv(ret_path + rf'\stock_ret.csv', encoding='utf-8')
        print('成功从wind获取股票半年期收益数据并保存本地！')
        return df
# stock_ret= get_stock_ret(start_dt=start_date, end_dt=end_date, struc_path=used_data_path, ret_path=used_data_path)
# print(stock_ret)


# 获取基准/基金的半年期收益率，index +1 天到下一个index的区间内, index为date，最后一个index为end_dt
def get_ret_semiyear(start_dt, end_dt, wcode, file_path, bench=False, fund=False):
    # param: wcode可为单个wcode，也可为[]wcode
    # 半年期日期
    dt_semi = date_semi(start_dt, end_dt)
    dt_semi = dt_semi.append(pd.DatetimeIndex([pd.to_datetime(end_dt)]))

    wcode = [wcode] if type(wcode) != list else wcode
    try:
        for code in wcode:
            df = pd.read_csv(file_path + rf'\{code}_ret_semiyear.csv', parse_dates=True, index_col=0)
            print(f'正在从本地获取{code}半年期收益文件：')

            # 更新指数收益数据
            if df.index[-1] < dt_semi[-1]:
                print(f'正在更新{code}半年期收益数据:')
                dt_semi_more = [dt for dt in dt_semi if dt not in df.index]
                df1 = pd.DataFrame(index=dt_semi_more, columns=df.columns)

                # 更新收益数据
                for i, dt in enumerate(dt_semi_more):
                    if dt != dt_semi_more[-1]:
                        dt_next = dt_semi_more[i + 1]
                        if bench:
                            df.loc[dt] = w.wss(code, "pct_chg_per", f"startDate={dt + relativedelta(days=1)};endDate={dt_next};ShowBlank=0").Data[0]
                        elif fund:
                            df.loc[dt] = w.wss(code, "NAV_adj_return", f"startDate={dt + relativedelta(days=1)};endDate={dt_next};ShowBlank=0").Data[0]
                df1.index.name = 'date'
                df = pd.concat([df, df1])
                df = df.dropna(how='all')  # 删掉没有数据的行
                df.index = pd.to_datetime(df.index)
                df = df[~df.index.duplicated(keep='first')]
                df.to_csv(file_path + rf'\{code}_ret_semiyear.csv')
                print(f'{code}每日收益数据更新完毕！')
        return df

    except FileNotFoundError:
        print(f'正在从wind获取{wcode}半年期收益数据：')
        df = pd.DataFrame(index=dt_semi, columns=wcode)
        for i, dt in enumerate(dt_semi):
            if dt != dt_semi[-1]:
                dt_next = dt_semi[i+1]
                if bench:
                    df.loc[dt] = w.wss(wcode, "pct_chg_per", f"startDate={dt + relativedelta(days=1)};endDate={dt_next};ShowBlank=0").Data[0]
                elif fund:
                    df.loc[dt] = w.wss(wcode, "NAV_adj_return", f"startDate={dt + relativedelta(days=1)};endDate={dt_next};ShowBlank=0").Data[0]
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index)
        df = df.dropna(how='all')  # 删掉没有数据的行
        for col in df.columns:
            df[[col]].to_csv(file_path + rf'\{col}_ret_semiyear.csv')
            print(f'成功从wind获取{col}每日收益数据并保存本地！')
        return df
# index_ret = get_ret_semiyear(start_dt='2023-01-01', end_dt='2024-04-01', wcode='000004.SH', file_path=semiyear_ret_data_path, bench=True)
# print(index_ret)
#
# fund_ret = get_ret_semiyear(start_dt=start_date, end_dt=end_date, wcode=[fund_code], file_path=semiyear_ret_data_path, fund=True)
# print(fund_ret)


# 获取基准/基金日历日日频收益率 当天
def get_ret_daily(start_dt, end_dt, wcode, file_path, bench=False, fund=False):
    # param: wcode 可为单个code，也可为[]code
    wcode = [wcode] if type(wcode) != list else wcode
    try:
        for code in wcode:
            df = pd.read_csv(file_path + rf'\{code}_ret_daily.csv', parse_dates=True, index_col=0)
            print(f'正在从本地获取{code}每日收益文件：')

            # 更新指数收益数据
            if df.index[-1] < datetime.strptime(end_dt, '%Y-%m-%d'):
                print(f'正在更新{code}每日收益数据:')
                dt_daily_more = w.tdays(df.index[-1], end_dt, "Days=Alldays").Data[0]

                # 更新收益数据
                if fund:
                    # 日历日，无值填充0，后复权基金收益率
                    tmp = w.wsd(code, "NAV_adj_return1", f"{dt_daily_more[0]}", f"{dt_daily_more[-1]}",
                                "Days=Alldays;ShowBlank=0;PriceAdj=B")
                if bench:
                    # 日历日，无值填充0，不复权基准收益率
                    tmp = w.wsd(code, "pct_chg", f"{dt_daily_more[0]}", f"{dt_daily_more[-1]}", "Days=Alldays;ShowBlank=0")
                df1 = pd.DataFrame(np.array(tmp.Data).T, index=tmp.Times, columns=tmp.Codes)
                df1.index.name = 'date'

                df = pd.concat([df, df1])
                df = df.dropna(how='all')  # 删掉没有数据的行
                df.index = pd.to_datetime(df.index)
                df = df[~df.index.duplicated(keep='first')]
                for col in df.columns:
                    df[[col]].to_csv(file_path + rf'\{col}_ret_daily.csv')
                    print(f'{col}每日收益数据更新完毕！')
        return df

    except FileNotFoundError:
        print(f'正在从wind获取{wcode}每日收益数据：')
        dt_daily = w.tdays(start_dt, end_dt, "Days=Alldays").Data[0]
        if fund:
            # 日历日，无值填充0，后复权基金收益率
            tmp = w.wsd(wcode, "NAV_adj_return1", f"{dt_daily[0]}", f"{dt_daily[-1]}", "Days=Alldays;ShowBlank=0;PriceAdj=B")
        if bench:
            # 日历日，无值填充0，不复权基准收益率
            tmp = w.wsd(wcode, "pct_chg", f"{dt_daily[0]}", f"{dt_daily[-1]}", "Days=Alldays;ShowBlank=0")
        df = pd.DataFrame(np.array(tmp.Data).T, index=tmp.Times, columns=tmp.Codes)
        df.index.name = 'date'
        df.index = pd.to_datetime(df.index)
        df = df.dropna(how='all')  # 删掉没有数据的行
        for col in df.columns:
            df[[col]].to_csv(file_path + rf'\{col}_ret_daily.csv')
            print(f'成功从wind获取{col}每日收益数据并保存本地！')
        return df
# index_ret = get_ret_daily(start_dt='2024-05-30', end_dt='2024-06-06', wcode=['000001.SH', '000002.SH'], file_path=daily_ret_data_path, bench=True)
# print(index_ret)
#
# fund_ret = get_ret_daily(start_dt='2024-05-30', end_dt='2024-06-10', wcode='005660.OF', file_path=daily_ret_data_path, fund=True)
# print(fund_ret)



















