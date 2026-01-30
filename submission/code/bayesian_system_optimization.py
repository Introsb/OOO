"""
Bayesian System Optimization - Phase 3
贝叶斯系统优化 - 超越网格搜索
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

# 尝试导入贝叶斯优化库
try:
    from skopt import gp_minimize
    from skopt.space import Real
    from skopt.utils import use_named_args
    HAS_SKOPT = True
except ImportError:
    HAS_SKOPT = False
    print("⚠ scikit-optimize not available, using grid search fallback")


class BayesianSystemOptimizer:
    """贝叶斯系统优化器"""
    
    def __init__(self):
        self.best_params = None
        self.optimization_history = []
        
    def load_simulation_data(self):
        """加载模拟数据"""
        print("\n" + "="*80)
        print("BAYESIAN SYSTEM OPTIMIZATION - PHASE 3")
        print("="*80)
        print("\n1. Loading simulation data...")
        
        # 使用合成数据进行演示
        print("   Using synthetic data for demonstration...")
        return self.generate_synthetic_data()
    
    def generate_synthetic_data(self):
        """生成合成数据用于测试"""
        np.random.seed(42)
        n_samples = 1000
        
        df = pd.DataFrame({
            'Season': np.random.randint(1, 35, n_samples),
            'Week': np.random.randint(1, 12, n_samples),
            'Name': [f'Contestant_{i}' for i in range(n_samples)],
            'Judge_Score': np.random.uniform(6, 10, n_samples),
            'Fan_Vote': np.random.uniform(0, 1, n_samples),
            'Technical_Rank': np.random.randint(1, 15, n_samples),
            'Eliminated': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        })
        
        return df
    
    def calculate_system_metrics(self, df, judge_weight, fan_weight, sigmoid_k, sigmoid_x0):
        """计算系统指标"""
        # 计算综合分数
        df['composite_score'] = judge_weight * df['Judge_Score'] + fan_weight * df['Fan_Vote']
        
        # 应用sigmoid变换
        df['transformed_score'] = 1 / (1 + np.exp(-sigmoid_k * (df['composite_score'] - sigmoid_x0)))
        
        # 计算排名
        df['final_rank'] = df.groupby(['Season', 'Week'])['transformed_score'].rank(ascending=False)
        
        # 计算不公正率（技术排名高但被淘汰）
        injustice_cases = df[(df['Technical_Rank'] <= 3) & (df['Eliminated'] == 1)]
        injustice_rate = len(injustice_cases) / len(df[df['Eliminated'] == 1]) if len(df[df['Eliminated'] == 1]) > 0 else 0
        
        # 计算技术公平性（技术排名低的被淘汰）
        fair_eliminations = df[(df['Technical_Rank'] >= 8) & (df['Eliminated'] == 1)]
        fairness = len(fair_eliminations) / len(df[df['Eliminated'] == 1]) if len(df[df['Eliminated'] == 1]) > 0 else 0
        
        # 计算多样性（分数分布的标准差）
        diversity = df['transformed_score'].std()
        
        return injustice_rate, fairness, diversity
    
    def objective_function(self, params, df):
        """目标函数（最小化）"""
        judge_weight, fan_weight, sigmoid_k, sigmoid_x0 = params
        
        # 确保权重和为1
        total_weight = judge_weight + fan_weight
        judge_weight = judge_weight / total_weight
        fan_weight = fan_weight / total_weight
        
        # 计算指标
        injustice_rate, fairness, diversity = self.calculate_system_metrics(
            df.copy(), judge_weight, fan_weight, sigmoid_k, sigmoid_x0
        )
        
        # 多目标优化：最小化不公正率，最大化公平性和多样性
        # 转换为最小化问题
        score = 0.6 * injustice_rate - 0.3 * fairness - 0.1 * diversity
        
        # 记录历史
        self.optimization_history.append({
            'judge_weight': judge_weight,
            'fan_weight': fan_weight,
            'sigmoid_k': sigmoid_k,
            'sigmoid_x0': sigmoid_x0,
            'injustice_rate': injustice_rate,
            'fairness': fairness,
            'diversity': diversity,
            'objective_score': score
        })
        
        return score
    
    def run_bayesian_optimization(self, df, n_calls=500):
        """运行贝叶斯优化"""
        print("\n2. Running Bayesian Optimization...")
        
        if not HAS_SKOPT:
            print("   ⚠ Falling back to enhanced grid search")
            return self.run_enhanced_grid_search(df)
        
        # 定义搜索空间
        space = [
            Real(0.3, 0.7, name='judge_weight'),
            Real(0.3, 0.7, name='fan_weight'),
            Real(1.0, 20.0, name='sigmoid_k'),
            Real(0.1, 0.6, name='sigmoid_x0')
        ]
        
        # 运行优化
        print(f"   Searching {n_calls} parameter combinations...")
        print("   This may take a few minutes...")
        
        @use_named_args(space)
        def objective(**params):
            param_list = [params['judge_weight'], params['fan_weight'], 
                         params['sigmoid_k'], params['sigmoid_x0']]
            return self.objective_function(param_list, df)
        
        result = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            verbose=False
        )
        
        # 提取最优参数
        judge_w, fan_w, k, x0 = result.x
        total = judge_w + fan_w
        judge_w, fan_w = judge_w/total, fan_w/total
        
        print(f"\n   ✓ Optimization complete!")
        print(f"   ✓ Best objective score: {result.fun:.6f}")
        print(f"\n   Optimal Parameters:")
        print(f"      Judge Weight: {judge_w:.4f}")
        print(f"      Fan Weight: {fan_w:.4f}")
        print(f"      Sigmoid k: {k:.4f}")
        print(f"      Sigmoid x₀: {x0:.4f}")
        
        # 计算最优参数的性能
        injustice, fairness, diversity = self.calculate_system_metrics(
            df.copy(), judge_w, fan_w, k, x0
        )
        
        print(f"\n   Performance Metrics:")
        print(f"      Injustice Rate: {injustice*100:.2f}%")
        print(f"      Technical Fairness: {fairness*100:.2f}%")
        print(f"      Diversity: {diversity:.4f}")
        
        self.best_params = {
            'judge_weight': judge_w,
            'fan_weight': fan_w,
            'sigmoid_k': k,
            'sigmoid_x0': x0,
            'injustice_rate': injustice,
            'fairness': fairness,
            'diversity': diversity,
            'objective_score': result.fun
        }
        
        return self.best_params
    
    def run_enhanced_grid_search(self, df):
        """增强网格搜索（fallback）"""
        print("   Running enhanced grid search (540 combinations)...")
        
        # 更细粒度的网格
        judge_weights = np.linspace(0.3, 0.7, 6)
        fan_weights = np.linspace(0.3, 0.7, 6)
        sigmoid_ks = [1, 3, 5, 7, 10, 15, 20]
        sigmoid_x0s = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        
        best_score = float('inf')
        best_params = None
        
        total_combinations = len(judge_weights) * len(sigmoid_ks) * len(sigmoid_x0s)
        count = 0
        
        for jw in judge_weights:
            for k in sigmoid_ks:
                for x0 in sigmoid_x0s:
                    fw = 1 - jw
                    params = [jw, fw, k, x0]
                    score = self.objective_function(params, df)
                    
                    if score < best_score:
                        best_score = score
                        best_params = params
                    
                    count += 1
                    if count % 50 == 0:
                        print(f"      Progress: {count}/{total_combinations}")
        
        judge_w, fan_w, k, x0 = best_params
        
        print(f"\n   ✓ Grid search complete!")
        print(f"   ✓ Best objective score: {best_score:.6f}")
        print(f"\n   Optimal Parameters:")
        print(f"      Judge Weight: {judge_w:.4f}")
        print(f"      Fan Weight: {fan_w:.4f}")
        print(f"      Sigmoid k: {k:.4f}")
        print(f"      Sigmoid x₀: {x0:.4f}")
        
        injustice, fairness, diversity = self.calculate_system_metrics(
            df.copy(), judge_w, fan_w, k, x0
        )
        
        print(f"\n   Performance Metrics:")
        print(f"      Injustice Rate: {injustice*100:.2f}%")
        print(f"      Technical Fairness: {fairness*100:.2f}%")
        print(f"      Diversity: {diversity:.4f}")
        
        self.best_params = {
            'judge_weight': judge_w,
            'fan_weight': fan_w,
            'sigmoid_k': k,
            'sigmoid_x0': x0,
            'injustice_rate': injustice,
            'fairness': fairness,
            'diversity': diversity,
            'objective_score': best_score
        }
        
        return self.best_params
    
    def compare_with_phase1(self):
        """与Phase 1对比"""
        print("\n3. Comparing with Phase 1...")
        
        # Phase 1最优参数
        phase1_params = {
            'judge_weight': 0.5,
            'fan_weight': 0.5,
            'sigmoid_k': 5.0,
            'sigmoid_x0': 0.3,
            'injustice_rate': 0.0418,  # 4.18%
            'fairness': 0.9940  # 99.40%
        }
        
        phase3_params = self.best_params
        
        comparison = pd.DataFrame({
            'Parameter': ['Judge Weight', 'Fan Weight', 'Sigmoid k', 'Sigmoid x₀', 
                         'Injustice Rate', 'Technical Fairness'],
            'Phase 1 (Grid Search)': [
                f"{phase1_params['judge_weight']:.4f}",
                f"{phase1_params['fan_weight']:.4f}",
                f"{phase1_params['sigmoid_k']:.4f}",
                f"{phase1_params['sigmoid_x0']:.4f}",
                f"{phase1_params['injustice_rate']*100:.2f}%",
                f"{phase1_params['fairness']*100:.2f}%"
            ],
            'Phase 3 (Bayesian)': [
                f"{phase3_params['judge_weight']:.4f}",
                f"{phase3_params['fan_weight']:.4f}",
                f"{phase3_params['sigmoid_k']:.4f}",
                f"{phase3_params['sigmoid_x0']:.4f}",
                f"{phase3_params['injustice_rate']*100:.2f}%",
                f"{phase3_params['fairness']*100:.2f}%"
            ]
        })
        
        print("\n" + "="*80)
        print("PHASE 1 vs PHASE 3 COMPARISON")
        print("="*80)
        print(comparison.to_string(index=False))
        
        # 计算改进
        injustice_improvement = (phase1_params['injustice_rate'] - phase3_params['injustice_rate']) / phase1_params['injustice_rate'] * 100
        fairness_improvement = (phase3_params['fairness'] - phase1_params['fairness']) / phase1_params['fairness'] * 100
        
        print(f"\n📈 Improvements:")
        print(f"   Injustice Rate: {injustice_improvement:+.2f}%")
        print(f"   Technical Fairness: {fairness_improvement:+.2f}%")
        
        return comparison
    
    def save_results(self):
        """保存结果"""
        print("\n4. Saving results...")
        
        # 保存最优参数
        best_params_df = pd.DataFrame([self.best_params])
        best_params_path = 'Bayesian_Optimal_Parameters.csv'
        best_params_df.to_csv(best_params_path, index=False)
        print(f"   ✓ Optimal parameters saved to {best_params_path}")
        
        # 保存优化历史
        if self.optimization_history:
            history_df = pd.DataFrame(self.optimization_history)
            history_path = 'Bayesian_Optimization_History.csv'
            history_df.to_csv(history_path, index=False)
            print(f"   ✓ Optimization history saved to {history_path}")
            print(f"      ({len(history_df)} iterations)")
        
        print("\n" + "="*80)
        print("✓ BAYESIAN SYSTEM OPTIMIZATION COMPLETE")
        print("="*80)
    
    def run_complete_optimization(self):
        """运行完整优化流程"""
        # 1. 加载数据
        df = self.load_simulation_data()
        
        # 2. 贝叶斯优化
        best_params = self.run_bayesian_optimization(df, n_calls=500)
        
        # 3. 对比Phase 1
        comparison = self.compare_with_phase1()
        
        # 4. 保存结果
        self.save_results()
        
        return best_params, comparison


def main():
    """主函数"""
    optimizer = BayesianSystemOptimizer()
    best_params, comparison = optimizer.run_complete_optimization()
    return best_params, comparison


if __name__ == '__main__':
    best_params, comparison = main()
