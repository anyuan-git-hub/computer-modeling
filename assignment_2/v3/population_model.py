"""
中国人口增长模型分析实验
=====================================
本实验使用1949-2023年中国人口数据,建立并对比三种经典人口增长模型:
1. 指数增长模型 (Exponential Growth Model)
2. 改进指数增长模型 (Improved Exponential Model)
3. Logistic增长模型 (Logistic Growth Model)

实验内容:
1. 模型构建: 建立三种人口增长模型
2. 参数估计: 使用多种方法进行参数估计
3. 模型检验: 留出数据法(train/test split)进行拟合度检验
4. 增长预测: 对未来人口进行预测
5. 结果对比: 从拟合精度、预测合理性、模型稳定性角度对比
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares, minimize
from scipy.stats import pearsonr
import openpyxl
import warnings
import time
import os
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')


def str_width(s):
    """计算字符串在终端中的显示宽度（中文=2，英文/数字=1）"""
    width = 0
    for c in str(s):
        if '\u4e00' <= c <= '\u9fff':
            width += 2
        else:
            width += 1
    return width


def fmt_col(s, width):
    """格式化字符串到指定显示宽度"""
    w = str_width(s)
    padding = width - w
    return str(s) + ' ' * max(0, padding)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """加载数据"""
    wb = openpyxl.load_workbook(r'd:\我的dd\课程作业-大二下\计算机建模\assignment_2\1949-2023人口数据-实验1.xlsx')
    sheet = wb['Sheet1']
    years, populations = [], []
    for row in sheet.iter_rows(min_row=6, values_only=True):
        if row[0] and row[1] and isinstance(row[0], (int, float)):
            year = int(row[0])
            pop = row[1]
            if isinstance(pop, str):
                pop = float(pop.replace('\xa0', '').replace(',', ''))
            else:
                pop = float(pop)
            years.append(year)
            populations.append(pop)
    return np.array(years), np.array(populations)


def mse(y_true, y_pred):
    """计算均方误差"""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """计算均方根误差"""
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    """计算平均绝对误差"""
    return np.mean(np.abs(y_true - y_pred))


def r_squared(y_true, y_pred):
    """计算决定系数R^2"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


def mape(y_true, y_pred):
    """计算平均绝对百分比误差"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# 人口增长模型定义
class PopulationModel:
    """人口增长模型基类"""
    pass


class ExponentialModel(PopulationModel):
    """指数增长模型: N(t) = N0 * e^(r*t)"""
    name = "指数增长"
    formula = "N(t) = N0 * exp(r * t)"
    param_names = ["N0", "r"]
    param_desc = {"N0": "初始人口(亿人)", "r": "年均增长率"}
    n_params = 2

    @staticmethod
    def model_func(t, N0, r):
        return N0 * np.exp(r * t)

    @staticmethod
    def initial_guess():
        return [5.4, 0.015]

    @staticmethod
    def bounds():
        return ([1, 0.001], [20, 0.1])


class ImprovedExponentialModel(PopulationModel):
    """改进指数增长模型: N(t) = N0 * e^(r*t) * (1 + b*t)"""
    name = "改进指数"
    formula = "N(t) = N0 * exp(r*t) * (1+b*t)"
    param_names = ["N0", "r", "b"]
    param_desc = {"N0": "初始人口(亿人)", "r": "基础增长率", "b": "修正因子"}
    n_params = 3

    @staticmethod
    def model_func(t, N0, r, b):
        return N0 * np.exp(r * t) * (1 + b * t)

    @staticmethod
    def initial_guess():
        return [5.4, 0.015, 0.001]

    @staticmethod
    def bounds():
        return ([1, 0.001, -0.01], [20, 0.1, 0.01])


class LogisticModel(PopulationModel):
    """Logistic增长模型: N(t) = K / (1 + (K/N0-1) * e^(-r*t))"""
    name = "Logistic"
    formula = "N(t) = K / (1 + (K/N0-1)*exp(-r*t))"
    param_names = ["N0", "r", "K"]
    param_desc = {"N0": "初始人口(亿人)", "r": "固有增长率", "K": "最大容量(亿人)"}
    n_params = 3

    @staticmethod
    def model_func(t, N0, r, K):
        return K / (1 + (K/N0 - 1) * np.exp(-r * t))

    @staticmethod
    def initial_guess():
        return [5.4, 0.03, 16]

    @staticmethod
    def bounds():
        return ([1, 0.001, 10], [20, 0.1, 25])


class GompertzModel(PopulationModel):
    """
    Gompertz增长模型: N(t) = K * exp(-b * e^(-r*t))

    模型特点:
        - 早期增长快于Logistic模型
        - 后期趋于稳定,但速度不同
        - 常用于描述生物生长过程
    """
    name = "Gompertz"
    formula = "N(t) = K * exp(-b * exp(-r*t))"
    param_names = ["K", "b", "r"]
    param_desc = {"K": "最大容量(亿人)", "b": "位移参数", "r": "增长率"}
    n_params = 3

    @staticmethod
    def model_func(t, K, b, r):
        return K * np.exp(-b * np.exp(-r * t))

    @staticmethod
    def initial_guess():
        return [15, 3, 0.05]

    @staticmethod
    def bounds():
        return ([5, 0.1, 0.01], [25, 10, 0.2])


class VonBertalanffyModel(PopulationModel):
    """
    Von Bertalanffy生长模型: N(t) = K * (1 - e^(-r*(t-t0)))^3

    模型特点:
        - 源于生物学中的体长生长研究
        - 考虑生物个体大小限制
        - 增长曲线呈渐近线形式
    """
    name = "Von Bertalanffy"
    formula = "N(t) = K * (1 - exp(-r*(t-t0)))^3"
    param_names = ["K", "r", "t0"]
    param_desc = {"K": "最大容量(亿人)", "r": "生长率", "t0": "理论起点时间"}
    n_params = 3

    @staticmethod
    def model_func(t, K, r, t0):
        diff = 1 - np.exp(-r * (t - t0))
        return K * np.power(np.maximum(diff, 1e-10), 3)

    @staticmethod
    def initial_guess():
        return [20, 0.03, -5]

    @staticmethod
    def bounds():
        return ([5, 0.01, -20], [30, 0.1, 10])


MODELS = [ExponentialModel, ImprovedExponentialModel, LogisticModel, GompertzModel, VonBertalanffyModel]


# 参数估计方法定义
class EstimationMethod:
    """参数估计方法基类"""
    pass


class TrustRegionReflective(EstimationMethod):
    """信赖域反射算法 (Trust Region Reflective)"""
    name = "信赖域反射法(TRF)"

    @staticmethod
    def estimate(model_class, t, y):
        try:
            p0 = model_class.initial_guess()
            bounds = model_class.bounds()
            popt, pcov = curve_fit(
                model_class.model_func, t, y,
                p0=p0, bounds=bounds, maxfev=50000, method='trf'
            )
            return {"params": popt, "success": True, "message": "成功"}
        except Exception as e:
            return {"params": None, "success": False, "message": str(e)}


class LevenbergMarquardt(EstimationMethod):
    """Levenberg-Marquardt算法"""
    name = "Levenberg-Marquardt"

    @staticmethod
    def estimate(model_class, t, y):
        try:
            p0 = model_class.initial_guess()
            popt, pcov = curve_fit(
                model_class.model_func, t, y,
                p0=p0, maxfev=50000, method='lm'
            )
            return {"params": popt, "success": True, "message": "成功"}
        except Exception as e:
            return {"params": None, "success": False, "message": str(e)}


class DogboxMethod(EstimationMethod):
    """Dogbox算法"""
    name = "Dogbox信赖域法"

    @staticmethod
    def estimate(model_class, t, y):
        try:
            p0 = model_class.initial_guess()
            bounds = model_class.bounds()

            def residuals(params):
                return y - model_class.model_func(t, *params)

            result = least_squares(
                residuals, p0, bounds=bounds,
                method='dogbox', ftol=1e-12, xtol=1e-12, gtol=1e-12,
                max_nfev=50000
            )
            return {
                "params": result.x if result.success else None,
                "success": result.success,
                "message": result.message if hasattr(result, 'message') else ("成功" if result.success else "失败")
            }
        except Exception as e:
            return {"params": None, "success": False, "message": str(e)}


class BFGSMethod(EstimationMethod):
    """BFGS拟牛顿法"""
    name = "BFGS拟牛顿法"

    @staticmethod
    def estimate(model_class, t, y):
        try:
            p0 = model_class.initial_guess()
            bounds = model_class.bounds()

            def objective(params):
                pred = model_class.model_func(t, *params)
                return np.sum((y - pred) ** 2)

            result = minimize(
                objective, p0, method='L-BFGS-B',
                bounds=list(zip(bounds[0], bounds[1])),
                options={'maxiter': 50000, 'ftol': 1e-12, 'gtol': 1e-12}
            )
            return {
                "params": result.x if result.success else None,
                "success": result.success,
                "message": result.message if hasattr(result, 'message') else ("成功" if result.success else "失败")
            }
        except Exception as e:
            return {"params": None, "success": False, "message": str(e)}


METHODS = [TrustRegionReflective, LevenbergMarquardt, DogboxMethod, BFGSMethod]


# 模型检验与预测
def train_test_split(years, populations, train_end_year=2010):
    """
    划分训练集和测试集(留出数据法)
    训练集: 1949 - train_end_year
    测试集: train_end_year+1 - 2023
    """
    train_mask = years <= train_end_year
    test_mask = years > train_end_year
    return (years[train_mask], populations[train_mask],
            years[test_mask], populations[test_mask])


def evaluate_model(model_class, params, t_train, y_train, t_test, y_test):
    """评估模型训练集和测试集性能"""
    y_train_pred = model_class.model_func(t_train, *params)
    y_test_pred = model_class.model_func(t_test, *params)

    train_metrics = {
        "mse": mse(y_train, y_train_pred),
        "rmse": rmse(y_train, y_train_pred),
        "mae": mae(y_train, y_train_pred),
        "r2": r_squared(y_train, y_train_pred),
        "mape": mape(y_train, y_train_pred)
    }

    test_metrics = {
        "mse": mse(y_test, y_test_pred),
        "rmse": rmse(y_test, y_test_pred),
        "mae": mae(y_test, y_test_pred),
        "r2": r_squared(y_test, y_test_pred),
        "mape": mape(y_test, y_test_pred)
    }

    return train_metrics, test_metrics


def predict_future(model_class, params, future_years, base_year=1949):
    """预测未来人口"""
    t_future = future_years - base_year
    return model_class.model_func(t_future, *params)


def analyze_prediction_quality(model_class, params, future_years):
    """
    分析预测质量
    检查: 预测值合理性、增长率稳定性、是否出现不合理趋势
    """
    preds = predict_future(model_class, params, future_years)

    growth_rates = []
    for i in range(1, len(preds)):
        rate = (preds[i] - preds[i-1]) / preds[i-1] * 100
        growth_rates.append(rate)

    warnings = []
    if preds[-1] < preds[-2] or preds[-2] < preds[-3]:
        warnings.append("预测人口出现下降趋势")
    if preds[-1] > 20:
        warnings.append(f"2050年预测超过20亿: {preds[-1]:.2f}")
    if any(abs(g) > 30 for g in growth_rates):
        warnings.append(f"增长率异常: {growth_rates}")

    return {
        "predictions": {y: p for y, p in zip(future_years, preds)},
        "growth_rates": {f"{future_years[i]}->{future_years[i+1]}": r
                        for i, r in enumerate(growth_rates)},
        "warnings": warnings,
        "is_reasonable": len(warnings) == 0
    }


def analyze_stability(results_dict):
    """
    分析模型稳定性
    检查不同方法估计的参数变异系数
    """
    stability = {}

    first_method = list(results_dict.keys())[0]
    model_names = results_dict[first_method].keys()

    for model_name in model_names:
        params_by_method = {}
        for method_name, method_results in results_dict.items():
            res = method_results.get(model_name)
            if res and res["success"]:
                params_by_method[method_name] = res["params"]

        if len(params_by_method) < 2:
            continue

        param_arrays = {}
        n_params = len(list(params_by_method.values())[0])

        for i in range(n_params):
            values = [p[i] for p in params_by_method.values()]
            mean_val = np.mean(values)
            std_val = np.std(values)
            param_arrays[f"param_{i}"] = {
                "mean": mean_val,
                "std": std_val,
                "cv": std_val / abs(mean_val) * 100 if mean_val != 0 else 0
            }

        stability[model_name] = {
            "n_methods": len(params_by_method),
            "params": param_arrays,
            "avg_cv": np.mean([p["cv"] for p in param_arrays.values()])
        }

    return stability


# 主实验流程
def run_full_experiment():
    """执行完整实验流程"""

    print("="*80)
    print(" " * 25 + "中国人口增长模型分析实验")
    print("="*80)

    years, populations = load_data()
    pop_yi = populations / 10000.0
    t = years - 1949

    print(f"\n数据范围: {years[0]} - {years[-1]}年")
    print(f"数据点数: {len(years)}")
    print(f"人口范围: {pop_yi.min():.2f}亿 - {pop_yi.max():.2f}亿")

    print(f"\n实验配置:")
    print(f"  - 人口增长模型: {len(MODELS)} 种")
    for m in MODELS:
        print(f"      - {m.name}: {m.formula}")
    print(f"  - 参数估计方法: {len(METHODS)} 种")
    for m in METHODS:
        print(f"      - {m.name}")

    train_years, train_pop, test_years, test_pop = train_test_split(years, pop_yi, train_end_year=2010)
    t_train = train_years - 1949
    t_test = test_years - 1949

    print(f"\n模型检验 - 留出数据法:")
    print(f"  训练集: {train_years[0]} - {train_years[-1]}年 ({len(train_years)}个数据点)")
    print(f"  测试集: {test_years[0]} - {test_years[-1]}年 ({len(test_years)}个数据点)")

    future_years = np.array([2024, 2025, 2030, 2035, 2040, 2045, 2050])
    base_year = 1949

    results = {}

    for method in METHODS:
        method_name = method.name
        print(f"\n{'='*80}")
        print(f"参数估计方法: {method_name}")
        print(f"{'='*80}")

        method_results = {}

        for model in MODELS:
            print(f"\n--- {model.name} ---")

            est_start = time.time()
            est_result = method.estimate(model, t_train, train_pop)
            est_time = time.time() - est_start

            if not est_result["success"] or est_result["params"] is None:
                print(f"  参数估计失败: {est_result['message']}")
                method_results[model.name] = {"success": False, "model": model}
                continue

            params = est_result["params"]

            train_metrics, test_metrics = evaluate_model(
                model, params, t_train, train_pop, t_test, test_pop
            )

            future_pred = predict_future(model, params, future_years)
            pred_quality = analyze_prediction_quality(model, params, future_years)

            method_results[model.name] = {
                "success": True,
                "params": params,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "future_predictions": future_pred,
                "prediction_quality": pred_quality,
                "time": est_time,
                "model": model
            }

            param_str = ", ".join([f"{p:.6f}" for p in params])
            print(f"  参数: {param_str}")
            print(f"  训练集 - R^2: {train_metrics['r2']:.6f}, RMSE: {train_metrics['rmse']:.6f}")
            print(f"  测试集 - R^2: {test_metrics['r2']:.6f}, RMSE: {test_metrics['rmse']:.6f}")
            print(f"  预测2030: {future_pred[2]:.2f}亿, 2035: {future_pred[3]:.2f}亿, 2040: {future_pred[4]:.2f}亿")
            print(f"  预测2045: {future_pred[5]:.2f}亿, 2050: {future_pred[6]:.2f}亿")
            if not pred_quality["is_reasonable"]:
                for w in pred_quality["warnings"]:
                    print(f"  警告: {w}")
            print(f"  计算时间: {est_time:.4f}s")

        results[method_name] = method_results

    print("\n" + "="*80)
    print("预测合理性分析")
    print("="*80)

    for method_name, method_results in results.items():
        print(f"\n{method_name}:")
        for model_name, res in method_results.items():
            if not res["success"]:
                continue
            pq = res["prediction_quality"]
            print(f"  {model_name}:")
            for y, p in pq["predictions"].items():
                print(f"    {y}年: {p:.2f}亿")
            for period, rate in pq["growth_rates"].items():
                print(f"    {period}增长率: {rate:.2f}%")
            if not pq["is_reasonable"]:
                for w in pq["warnings"]:
                    print(f"    警告: {w}")

    print("\n" + "="*80)
    print("模型稳定性分析")
    print("="*80)

    stability = analyze_stability(results)

    for model_name, stab in stability.items():
        print(f"\n{model_name}:")
        print(f"  成功拟合方法数: {stab['n_methods']}")
        for param_name, pstats in stab["params"].items():
            print(f"    {param_name}: 均值={pstats['mean']:.4f}, 标准差={pstats['std']:.6f}, 变异系数={pstats['cv']:.2f}%")
        print(f"  平均变异系数: {stab['avg_cv']:.2f}%")

    print("\n" + "="*100)
    print("未来人口预测汇总表")
    print("="*100)

    print(f"\n预测年份: {future_years}")
    print("-"*100)

    year_col_width = 10
    col_widths = [30, 18] + [year_col_width] * len(future_years)
    header_parts = [fmt_col("方法", col_widths[0]), fmt_col("模型", col_widths[1])]
    for i, year in enumerate(future_years):
        header_parts.append(fmt_col(str(year), col_widths[2 + i]))
    print(' '.join(header_parts))
    print("-"*100)

    for method_name, method_results in results.items():
        first = True
        for model_name, res in method_results.items():
            if not res["success"]:
                continue
            row_parts = [fmt_col(method_name if first else '', col_widths[0]),
                        fmt_col(model_name, col_widths[1])]
            for i, year in enumerate(future_years):
                row_parts.append(fmt_col(f"{res['future_predictions'][i]:.2f}", col_widths[2 + i]))
            print(' '.join(row_parts))
            first = False

    print("-"*100)

    print("\n" + "="*100)
    print("测试集预测结果对比表")
    print("="*100)

    print(f"\n测试集年份: {test_years}")
    print("-"*100)

    col_widths2 = [30, 18, 14, 14, 14, 14]
    header_parts = [fmt_col("方法", col_widths2[0]), fmt_col("模型", col_widths2[1]),
                   fmt_col("训练R^2", col_widths2[2]), fmt_col("测试R^2", col_widths2[3]),
                   fmt_col("测试RMSE", col_widths2[4]), fmt_col("测试MAE", col_widths2[5])]
    print(' '.join(header_parts))
    print("-"*100)

    for method_name, method_results in results.items():
        first = True
        for model_name, res in method_results.items():
            if not res["success"]:
                continue
            tm = res["train_metrics"]
            test_m = res["test_metrics"]
            row_parts = [fmt_col(method_name if first else '', col_widths2[0]),
                        fmt_col(model_name, col_widths2[1]),
                        fmt_col(f"{tm['r2']:.6f}", col_widths2[2]),
                        fmt_col(f"{test_m['r2']:.6f}", col_widths2[3]),
                        fmt_col(f"{test_m['rmse']:.6f}", col_widths2[4]),
                        fmt_col(f"{test_m['mae']:.6f}", col_widths2[5])]
            print(' '.join(row_parts))
            first = False

    print("-"*100)

    return results, years, pop_yi, t, train_years, test_years, train_pop, test_pop, future_years


def plot_comprehensive_results(results, years, pop_yi, t, train_years, test_years, train_pop, test_pop, future_years):
    """为每种方法生成独立的综合结果图"""

    output_path = OUTPUT_DIR
    os.makedirs(output_path, exist_ok=True)

    color_palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
    line_style_palette = ['-', '--', '-.', ':', '-']

    for method_name, method_results in results.items():
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        safe_method_name = method_name.replace('(', '').replace(')', '').replace(' ', '_')

        fig.suptitle(f'{method_name}\n人口增长模型分析', fontsize=16, fontweight='bold')

        ax1 = axes[0, 0]
        ax1.scatter(train_years, train_pop, color='blue', s=50, label='训练集数据 (1949-2010)', zorder=5, alpha=0.7)
        ax1.scatter(test_years, test_pop, color='red', s=50, label='测试集数据 (2011-2023)', zorder=5, alpha=0.7)

        fit_end_year = 2023
        fit_end_t = fit_end_year - 1949
        t_fit = np.linspace(0, fit_end_t, 500)
        years_fit = t_fit + 1949

        model_names = []
        for idx, (model_name, res) in enumerate(method_results.items()):
            if not res["success"]:
                continue
            model_names.append(model_name)
            model = res["model"]
            params = res["params"]
            y_fit = model.model_func(t_fit, *params)
            ax1.plot(years_fit, y_fit, color=color_palette[idx % len(color_palette)],
                    linestyle=line_style_palette[idx % len(line_style_palette)], linewidth=2,
                    label=f'{model_name}')

        ax1.axvline(x=2010, color='gray', linestyle='--', alpha=0.7, label='训练/测试分界线')

        future_start_year = 2024
        future_start_t = future_start_year - 1949
        t_future = np.linspace(future_start_t, 110, 500)
        years_future = t_future + 1949
        for idx, (model_name, res) in enumerate(method_results.items()):
            if not res["success"]:
                continue
            model = res["model"]
            params = res["params"]
            y_future = model.model_func(t_future, *params)
            ax1.plot(years_future, y_future, color=color_palette[idx % len(color_palette)],
                    linestyle=line_style_palette[idx % len(line_style_palette)], linewidth=2, alpha=0.5)

        ax1.axvline(x=2023, color='orange', linestyle=':', alpha=0.7, label='历史/预测分界线')

        ax1.set_xlabel('年份', fontsize=12)
        ax1.set_ylabel('人口 (亿人)', fontsize=12)
        ax1.set_title('模型拟合与预测', fontsize=14)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(1945, 2060)
        ax1.set_ylim(4, 32)

        ax2 = axes[0, 1]
        model_names = []
        train_r2 = []
        test_r2 = []

        for model_name, res in method_results.items():
            if not res["success"]:
                continue
            model_names.append(model_name)
            train_r2.append(res["train_metrics"]["r2"])
            test_r2.append(res["test_metrics"]["r2"])

        x = np.arange(len(model_names))
        width = 0.35

        bars1 = ax2.bar(x - width/2, train_r2, width, label='训练集R^2', color='steelblue', alpha=0.8)
        bars2 = ax2.bar(x + width/2, test_r2, width, label='测试集R^2', color='coral', alpha=0.8)

        ax2.set_ylabel('R^2 分数', fontsize=12)
        ax2.set_title('训练集 vs 测试集 R^2', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_names, rotation=15, ha='right')
        ax2.legend()
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.grid(True, alpha=0.3, axis='y')

        for bar, val in zip(bars1, train_r2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=9)
        for bar, val in zip(bars2, test_r2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.1 if val < 0 else bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', fontsize=9)

        ax3 = axes[1, 0]

        comparison_years = [2030, 2040, 2050]
        n_years = len(comparison_years)
        n_models = len(model_names)
        x = np.arange(n_years)

        bar_width = 0.6 / n_models
        group_width = 0.6

        for midx, model_name in enumerate(model_names):
            res = method_results[model_name]
            if not res["success"]:
                continue
            future_pred = [res["prediction_quality"]["predictions"][y] for y in comparison_years]
            positions = x + (midx - n_models/2 + 0.5) * bar_width
            bars = ax3.bar(positions, future_pred, bar_width,
                   label=model_name, color=color_palette[midx % len(color_palette)], alpha=0.8)
            for bar, val in zip(bars, future_pred):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.2f}', ha='center', fontsize=8, rotation=45)

        ax3.set_xticks(x)
        ax3.set_xticklabels([str(y) for y in comparison_years])
        ax3.set_xlabel('年份', fontsize=12)
        ax3.set_ylabel('人口 (亿人)', fontsize=12)
        ax3.set_title('未来人口预测', fontsize=14)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')

        ax4 = axes[1, 1]
        ax4.axis('off')

        table_data = [["模型", "参数", "训练R^2", "测试R^2", "2050预测"]]

        for model_name, res in method_results.items():
            if not res["success"]:
                continue
            params_str = ", ".join([f"{p:.4f}" for p in res["params"]])
            tm = res["train_metrics"]
            test_m = res["test_metrics"]
            future = res["prediction_quality"]["predictions"][2050]

            table_data.append([
                model_name,
                params_str,
                f"{tm['r2']:.4f}",
                f"{test_m['r2']:.4f}",
                f"{future:.2f}"
            ])

        table = ax4.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc='center',
            cellLoc='center',
            colWidths=[0.13, 0.30, 0.13, 0.13, 0.13]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 2.0)

        for i in range(len(table_data[0])):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')

        for i in range(1, len(table_data)):
            for j in range(len(table_data[0])):
                if i % 2 == 0:
                    table[(i, j)].set_facecolor('#D6DCE4')
                else:
                    table[(i, j)].set_facecolor('#FFFFFF')

        ax4.set_title('结果汇总', fontsize=14, pad=20)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filepath = f"{output_path}/{safe_method_name}.png"
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"  [保存] {filepath}")

    print(f"\n所有图表已保存至: {output_path}")


if __name__ == "__main__":
    results, years, pop_yi, t, train_years, test_years, train_pop, test_pop, future_years = run_full_experiment()
    plot_comprehensive_results(results, years, pop_yi, t, train_years, test_years, train_pop, test_pop, future_years)

    print("\n" + "="*80)
    print("实验完成!")
    print("="*80)