#!/usr/bin/env python3
"""
NBA Betting Analyzer - Backend Avancé avec Régression Statistique
Inclut tests de significativité, R², p-values, et calcul de rendement attendu
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime, timedelta
import json
from betonline_scraper import BetOnlineScraper

app = Flask(__name__)
CORS(app)

class StatisticalNBAAnalyzer:
    def __init__(self):
        self.scraper = BetOnlineScraper()
        self.model = None
        self.regression_stats = {}
        
    def get_player_historical_data(self, player_name, n_games=20):
        """
        Récupère les données historiques d'un joueur
        Dans la vraie version: scrape Basketball-Reference
        """
        # Génère des données réalistes pour la démo
        np.random.seed(hash(player_name) % 2**32)
        
        base_points = np.random.uniform(22, 30)
        
        games = []
        for i in range(n_games):
            # Simule une variation réaliste
            points = base_points + np.random.normal(0, 4)
            
            game = {
                'game_num': i + 1,
                'date': (datetime.now() - timedelta(days=i*3)).strftime('%Y-%m-%d'),
                'opponent': self._get_random_team(),
                'is_home': np.random.choice([0, 1]),
                'points': max(0, points),
                'minutes': np.random.uniform(30, 38),
                'opponent_def_rating': np.random.uniform(105, 115),
                'team_pace': np.random.uniform(95, 105),
                'rest_days': np.random.choice([0, 1, 2, 3]),
                'back_to_back': np.random.choice([0, 1], p=[0.85, 0.15])
            }
            games.append(game)
        
        return pd.DataFrame(games)
    
    def _get_random_team(self):
        teams = ['Lakers', 'Celtics', 'Warriors', 'Nets', 'Heat', 'Bucks',
                'Suns', 'Mavericks', 'Nuggets', 'Clippers', '76ers', 'Raptors']
        return np.random.choice(teams)
    
    def build_regression_model(self, player_name):
        """
        Construit un modèle de régression avec toutes les variables explicatives
        """
        df = self.get_player_historical_data(player_name, n_games=20)
        
        # Variables indépendantes (X)
        X = df[['is_home', 'opponent_def_rating', 'minutes', 'rest_days', 'back_to_back', 'team_pace']]
        
        # Variable dépendante (Y)
        y = df['points']
        
        # Régression linéaire
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédictions
        y_pred = model.predict(X)
        
        # Calcul des statistiques
        residuals = y - y_pred
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # Calcul des p-values et tests de significativité
        n = len(y)
        k = X.shape[1]  # Nombre de variables
        dof = n - k - 1  # Degrés de liberté
        
        # Variance des résidus
        var_residuals = np.sum(residuals**2) / dof
        
        # Variance des coefficients
        var_coef = var_residuals * np.linalg.inv(X.T @ X).diagonal()
        std_errors = np.sqrt(var_coef)
        
        # T-statistiques et p-values
        t_stats = model.coef_ / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        
        # Intervalles de confiance à 95%
        t_critical = stats.t.ppf(0.975, dof)
        conf_intervals = [
            (coef - t_critical * se, coef + t_critical * se)
            for coef, se in zip(model.coef_, std_errors)
        ]
        
        # Stocke les statistiques
        self.regression_stats = {
            'intercept': {
                'value': float(model.intercept_),
                'interpretation': 'Points de base'
            },
            'coefficients': [
                {
                    'variable': 'is_home',
                    'coefficient': float(model.coef_[0]),
                    'std_error': float(std_errors[0]),
                    't_stat': float(t_stats[0]),
                    'p_value': float(p_values[0]),
                    'significant': p_values[0] < 0.05,
                    'conf_interval_95': [float(conf_intervals[0][0]), float(conf_intervals[0][1])],
                    'interpretation': f"{'Boost' if model.coef_[0] > 0 else 'Pénalité'} de {abs(model.coef_[0]):.2f} pts à domicile"
                },
                {
                    'variable': 'opponent_def_rating',
                    'coefficient': float(model.coef_[1]),
                    'std_error': float(std_errors[1]),
                    't_stat': float(t_stats[1]),
                    'p_value': float(p_values[1]),
                    'significant': p_values[1] < 0.05,
                    'conf_interval_95': [float(conf_intervals[1][0]), float(conf_intervals[1][1])],
                    'interpretation': f"{'Augmentation' if model.coef_[1] > 0 else 'Diminution'} de {abs(model.coef_[1]):.2f} pts par point de def rating"
                },
                {
                    'variable': 'minutes',
                    'coefficient': float(model.coef_[2]),
                    'std_error': float(std_errors[2]),
                    't_stat': float(t_stats[2]),
                    'p_value': float(p_values[2]),
                    'significant': p_values[2] < 0.05,
                    'conf_interval_95': [float(conf_intervals[2][0]), float(conf_intervals[2][1])],
                    'interpretation': f"{abs(model.coef_[2]):.3f} pts par minute jouée"
                },
                {
                    'variable': 'rest_days',
                    'coefficient': float(model.coef_[3]),
                    'std_error': float(std_errors[3]),
                    't_stat': float(t_stats[3]),
                    'p_value': float(p_values[3]),
                    'significant': p_values[3] < 0.05,
                    'conf_interval_95': [float(conf_intervals[3][0]), float(conf_intervals[3][1])],
                    'interpretation': f"{abs(model.coef_[3]):.2f} pts par jour de repos"
                },
                {
                    'variable': 'back_to_back',
                    'coefficient': float(model.coef_[4]),
                    'std_error': float(std_errors[4]),
                    't_stat': float(t_stats[4]),
                    'p_value': float(p_values[4]),
                    'significant': p_values[4] < 0.05,
                    'conf_interval_95': [float(conf_intervals[4][0]), float(conf_intervals[4][1])],
                    'interpretation': f"{'Pénalité' if model.coef_[4] < 0 else 'Boost'} de {abs(model.coef_[4]):.2f} pts en back-to-back"
                },
                {
                    'variable': 'team_pace',
                    'coefficient': float(model.coef_[5]),
                    'std_error': float(std_errors[5]),
                    't_stat': float(t_stats[5]),
                    'p_value': float(p_values[5]),
                    'significant': p_values[5] < 0.05,
                    'conf_interval_95': [float(conf_intervals[5][0]), float(conf_intervals[5][1])],
                    'interpretation': f"{abs(model.coef_[5]):.3f} pts par unité de pace"
                }
            ],
            'model_quality': {
                'r_squared': float(r2),
                'adjusted_r_squared': float(1 - (1 - r2) * (n - 1) / dof),
                'rmse': float(rmse),
                'mae': float(np.mean(np.abs(residuals))),
                'sample_size': int(n),
                'degrees_of_freedom': int(dof)
            }
        }
        
        self.model = model
        return model, self.regression_stats
    
    def predict_with_confidence(self, player_name, opponent, is_home, minutes=35, rest_days=1, back_to_back=0):
        """
        Fait une prédiction avec intervalle de confiance et calcul de rendement
        """
        # Construit le modèle
        model, reg_stats = self.build_regression_model(player_name)
        
        # Obtient la ligne BetOnline
        betonline_line = self.scraper.get_player_line(player_name)
        
        if betonline_line:
            bookmaker_line = betonline_line['line']
            over_odds = betonline_line['over_odds']
            under_odds = betonline_line['under_odds']
        else:
            # Fallback
            bookmaker_line = 25.5
            over_odds = '-110'
            under_odds = '-110'
        
        # Prépare les données pour la prédiction
        opponent_def_rating = np.random.uniform(105, 115)  # À remplacer par vraie data
        team_pace = np.random.uniform(95, 105)  # À remplacer par vraie data
        
        X_pred = np.array([[
            1 if is_home else 0,
            opponent_def_rating,
            minutes,
            rest_days,
            back_to_back,
            team_pace
        ]])
        
        # Prédiction
        predicted_points = float(model.predict(X_pred)[0])
        
        # Calcul de l'écart-type de la prédiction
        df = self.get_player_historical_data(player_name, n_games=20)
        y_train = df['points']
        y_train_pred = model.predict(df[['is_home', 'opponent_def_rating', 'minutes', 'rest_days', 'back_to_back', 'team_pace']])
        std_dev = np.std(y_train - y_train_pred)
        
        # Intervalle de confiance à 80%
        z_80 = 1.28
        conf_interval = {
            'low': float(predicted_points - z_80 * std_dev),
            'high': float(predicted_points + z_80 * std_dev),
            'std_dev': float(std_dev)
        }
        
        # Calcul de la probabilité que le joueur dépasse la ligne
        z_score = (predicted_points - bookmaker_line) / std_dev
        prob_over = float(stats.norm.cdf(z_score))
        prob_under = 1 - prob_over
        
        # Conversion des odds américains en probabilité implicite
        def american_to_prob(odds_str):
            odds = int(odds_str)
            if odds < 0:
                return abs(odds) / (abs(odds) + 100)
            else:
                return 100 / (odds + 100)
        
        implied_prob_over = american_to_prob(over_odds)
        implied_prob_under = american_to_prob(under_odds)
        
        # Calcul de l'edge
        edge_over = prob_over - implied_prob_over
        edge_under = prob_under - implied_prob_under
        
        # Calcul du rendement attendu (Kelly Criterion)
        def kelly_stake(prob, odds_str):
            odds = int(odds_str)
            if odds < 0:
                decimal_odds = 1 + (100 / abs(odds))
            else:
                decimal_odds = 1 + (odds / 100)
            
            q = 1 - prob
            kelly = (prob * decimal_odds - q) / decimal_odds
            return max(0, kelly * 0.25)  # Quarter Kelly
        
        kelly_over = kelly_stake(prob_over, over_odds)
        kelly_under = kelly_stake(prob_under, under_odds)
        
        # Rendement attendu
        expected_return_over = edge_over * 100
        expected_return_under = edge_under * 100
        
        # Décision
        if abs(edge_over) > abs(edge_under) and edge_over > 0.05:
            recommendation = 'OVER'
            edge = edge_over
            prob = prob_over
            kelly = kelly_over
            expected_return = expected_return_over
            confidence = int(prob * 100)
        elif edge_under > 0.05:
            recommendation = 'UNDER'
            edge = edge_under
            prob = prob_under
            kelly = kelly_under
            expected_return = expected_return_under
            confidence = int(prob * 100)
        else:
            recommendation = 'SKIP'
            edge = 0
            prob = 0.5
            kelly = 0
            expected_return = 0
            confidence = 50
        
        return {
            'player': player_name,
            'opponent': opponent,
            'is_home': is_home,
            'prediction': {
                'points': round(predicted_points, 2),
                'std_dev': round(std_dev, 2),
                'confidence_interval_80': conf_interval
            },
            'betonline': {
                'line': bookmaker_line,
                'over_odds': over_odds,
                'under_odds': under_odds
            },
            'probabilities': {
                'over': round(prob_over * 100, 1),
                'under': round(prob_under * 100, 1),
                'implied_over': round(implied_prob_over * 100, 1),
                'implied_under': round(implied_prob_under * 100, 1)
            },
            'recommendation': {
                'bet': recommendation,
                'edge': round(edge * 100, 2),
                'confidence': confidence,
                'kelly_stake': round(kelly * 100, 2),
                'expected_return': round(expected_return, 2)
            },
            'regression_stats': reg_stats
        }
    
    def get_all_opportunities(self, min_edge=5.0):
        """
        Scanne toutes les opportunités disponibles sur BetOnline
        """
        props = self.scraper.get_nba_player_props()
        opportunities = []
        
        for prop in props:
            try:
                result = self.predict_with_confidence(
                    prop['player'],
                    prop['opponent'],
                    prop['is_home']
                )
                
                # Ne garde que les opportunités avec edge minimum
                if result['recommendation']['edge'] >= min_edge:
                    opportunities.append(result)
                    
            except Exception as e:
                print(f"Erreur pour {prop['player']}: {e}")
                continue
        
        # Trie par edge décroissant
        opportunities.sort(key=lambda x: x['recommendation']['edge'], reverse=True)
        
        return opportunities

analyzer = StatisticalNBAAnalyzer()

@app.route('/api/analyze', methods=['POST'])
def analyze_player():
    """Analyse un joueur spécifique avec régression complète"""
    data = request.json
    player = data.get('player')
    opponent = data.get('opponent')
    is_home = data.get('is_home', True)
    
    result = analyzer.predict_with_confidence(player, opponent, is_home)
    return jsonify(result)

@app.route('/api/opportunities', methods=['GET'])
def get_opportunities():
    """Obtient toutes les opportunités avec edge minimum"""
    min_edge = request.args.get('min_edge', 5.0, type=float)
    opportunities = analyzer.get_all_opportunities(min_edge)
    return jsonify(opportunities)

@app.route('/api/betonline/props', methods=['GET'])
def get_betonline_props():
    """Obtient les props disponibles sur BetOnline"""
    scraper = BetOnlineScraper()
    props = scraper.get_nba_player_props()
    return jsonify(props)

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})

if __name__ == '__main__':
    print("🏀 NBA Statistical Analyzer - Backend Avancé")
    print("=" * 60)
    print("✅ Régression linéaire multiple")
    print("✅ Tests de significativité (p-values)")
    print("✅ Intervalles de confiance à 95%")
    print("✅ Calcul du rendement attendu (Kelly)")
    print("✅ Intégration BetOnline en temps réel")
    print("=" * 60)
    print("🌐 Server: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
