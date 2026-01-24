#!/usr/bin/env python3
"""
NBA Betting Analyzer - Vercel Serverless Entry Point
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import sys
import os

# Ajouter le dossier parent au path pour importer les modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

app = Flask(__name__)
CORS(app)

# Import nba_api si disponible
try:
    from nba_api.stats.static import players
    from nba_api.stats.endpoints import playergamelog
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

class NBAAnalyzer:
    """Analyseur NBA simplifié pour Vercel"""
    
    def __init__(self):
        self.defensive_ratings = {
            'LAL': 112.3, 'GSW': 110.5, 'BOS': 108.2, 'MIA': 109.8,
            'MIL': 110.1, 'PHX': 111.4, 'DAL': 112.8, 'DEN': 109.2,
            'LAC': 110.7, 'PHI': 108.9, 'BKN': 114.2, 'ATL': 113.5,
            'CHI': 112.1, 'CLE': 109.5, 'DET': 115.3, 'HOU': 113.8,
            'IND': 114.5, 'MEM': 111.2, 'MIN': 110.4, 'NOP': 113.2,
            'NYK': 109.8, 'OKC': 108.5, 'ORL': 110.3, 'POR': 114.8,
            'SAC': 112.6, 'SAS': 115.1, 'TOR': 113.4, 'UTA': 114.2,
            'WAS': 116.5, 'CHA': 115.8
        }
    
    def get_player_data(self, player_name, n_games=20):
        """Récupère données joueur (simulées pour démo)"""
        np.random.seed(hash(player_name) % 2**32)
        base_points = np.random.uniform(20, 28)
        
        games = []
        for i in range(n_games):
            game = {
                'date': datetime.now().strftime('%Y-%m-%d'),
                'opponent': np.random.choice(['LAL', 'GSW', 'BOS', 'MIA']),
                'is_home': np.random.choice([True, False]),
                'points': base_points + np.random.normal(0, 4),
                'rebounds': np.random.uniform(4, 8),
                'assists': np.random.uniform(3, 7),
                'minutes': np.random.uniform(30, 38),
                'opponent_def_rating': np.random.uniform(108, 115),
                'rest_days': 1,
                'back_to_back': 0,
                'team_pace': 100.0,
                'result': np.random.choice(['W', 'L'])
            }
            games.append(game)
        
        return pd.DataFrame(games)
    
    def predict(self, player_name, opponent='LAL', is_home=True, line=None):
        """Fait une prédiction complète"""
        df = self.get_player_data(player_name, n_games=20)
        
        # Régression
        X = df[['is_home', 'opponent_def_rating', 'minutes', 'rest_days', 'back_to_back', 'team_pace']]
        y = df['points']
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédiction
        opp_def = self.defensive_ratings.get(opponent, 112.0)
        X_pred = np.array([[1 if is_home else 0, opp_def, 35, 1, 0, 100]])
        predicted_points = float(model.predict(X_pred)[0])
        std_dev = float(df['points'].std())
        
        # Stats
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        # Intervalle de confiance
        z_80 = 1.28
        conf_low = predicted_points - z_80 * std_dev
        conf_high = predicted_points + z_80 * std_dev
        
        # Ligne bookmaker
        if line is None:
            line = df['points'].mean() - 0.5
        
        # Probabilités
        z_score = (predicted_points - line) / std_dev
        prob_over = float(stats.norm.cdf(z_score))
        prob_under = 1 - prob_over
        
        # Edge
        implied_prob = 0.5
        edge_over = prob_over - implied_prob
        edge_under = prob_under - implied_prob
        
        # Recommandation
        if edge_over > 0.05 and edge_over > edge_under:
            recommendation = 'OVER'
            edge = edge_over
            confidence = int(prob_over * 100)
        elif edge_under > 0.05:
            recommendation = 'UNDER'
            edge = edge_under
            confidence = int(prob_under * 100)
        else:
            recommendation = 'SKIP'
            edge = 0
            confidence = 50
        
        return {
            'player': player_name,
            'opponent': opponent,
            'is_home': is_home,
            'data_source': 'Simulated for demo',
            'prediction': {
                'points': round(predicted_points, 2),
                'std_dev': round(std_dev, 2),
                'confidence_interval_80': {
                    'low': round(conf_low, 1),
                    'high': round(conf_high, 1)
                }
            },
            'bookmaker_line': round(line, 1),
            'probabilities': {
                'over': round(prob_over * 100, 1),
                'under': round(prob_under * 100, 1)
            },
            'recommendation': {
                'bet': recommendation,
                'edge': round(edge * 100, 2),
                'confidence': confidence
            },
            'regression_stats': {
                'model_quality': {
                    'r_squared': float(r2),
                    'rmse': float(rmse),
                    'sample_size': len(df)
                }
            },
            'recent_games': df[['date', 'opponent', 'points', 'result']].head(5).to_dict('records')
        }

# Initialize analyzer
analyzer = NBAAnalyzer()

# Routes
@app.route('/')
@app.route('/api')
def home():
    return jsonify({
        'service': 'NBA Betting Analyzer API',
        'version': '1.0.0',
        'status': 'active',
        'nba_api_available': NBA_API_AVAILABLE,
        'endpoints': {
            'analyze': 'POST /api/analyze',
            'health': 'GET /api/health'
        }
    })

@app.route('/api/health')
def health():
    return jsonify({
        'status': 'ok',
        'nba_api': NBA_API_AVAILABLE,
        'data_source': 'REAL' if NBA_API_AVAILABLE else 'SIMULATED',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze():
    """Analyse un joueur spécifique"""
    data = request.json
    player = data.get('player', 'LeBron James')
    opponent = data.get('opponent', 'GSW')
    is_home = data.get('is_home', True)
    line = data.get('line')
    
    result = analyzer.predict(player, opponent, is_home, line)
    return jsonify(result)

@app.route('/api/opportunities')
def opportunities():
    """Récupère les meilleures opportunités"""
    top_players = ['LeBron James', 'Stephen Curry', 'Luka Doncic']
    opps = []
    
    for player in top_players:
        try:
            result = analyzer.predict(player, 'GSW', True)
            if result['recommendation']['edge'] >= 5.0:
                opps.append(result)
        except:
            continue
    
    opps.sort(key=lambda x: x['recommendation']['edge'], reverse=True)
    return jsonify(opps)

# Vercel needs this
if __name__ == '__main__':
    app.run()