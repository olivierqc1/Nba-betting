#!/usr/bin/env python3
"""
NBA BETTING ANALYZER - VERSION COMPLÈTE GRATUITE
Système complet avec VRAIES données NBA via nba_api

Installation: pip install nba-api flask flask-cors pandas numpy scipy scikit-learn --break-system-packages
Usage: python3 nba_complete.py
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
import time

# Import nba_api
try:
    from nba_api.stats.static import players
    from nba_api.stats.endpoints import playergamelog
    NBA_API_AVAILABLE = True
except ImportError:
    print("⚠️ nba_api non installé. Installe avec: pip install nba-api --break-system-packages")
    NBA_API_AVAILABLE = False

app = Flask(__name__)
CORS(app)

class CompleteNBAAnalyzer:
    """
    Système complet d'analyse NBA avec vraies données gratuites
    """
    
    def __init__(self):
        self.cache = {}
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
        
        print("=" * 70)
        print("🏀 NBA BETTING ANALYZER - VERSION COMPLÈTE GRATUITE")
        print("=" * 70)
        if NBA_API_AVAILABLE:
            print("✅ nba_api installé - Utilise VRAIES DONNÉES")
        else:
            print("⚠️ nba_api non installé - Utilise données simulées")
            print("   Installe avec: pip install nba-api --break-system-packages")
        print("=" * 70)
    
    def find_player_id(self, player_name):
        """Trouve l'ID NBA d'un joueur"""
        if not NBA_API_AVAILABLE:
            return None
        
        player = players.find_players_by_full_name(player_name)
        if not player:
            return None
        return player[0]['id']
    
    def get_real_player_data(self, player_name, n_games=20):
        """
        Récupère les VRAIES données d'un joueur via nba_api
        """
        if not NBA_API_AVAILABLE:
            return None
        
        # Check cache
        cache_key = f"{player_name}_2024-25"
        if cache_key in self.cache:
            return self.cache[cache_key].head(n_games)
        
        player_id = self.find_player_id(player_name)
        if not player_id:
            print(f"⚠️ Joueur '{player_name}' non trouvé")
            return None
        
        try:
            print(f"🔍 Récupération des vraies stats pour {player_name}...")
            time.sleep(0.6)  # Rate limiting
            
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season='2024-25'
            )
            
            df = gamelog.get_data_frames()[0]
            
            if df.empty:
                return None
            
            # Reformate
            df_clean = pd.DataFrame({
                'date': pd.to_datetime(df['GAME_DATE']).dt.strftime('%Y-%m-%d'),
                'opponent': df['MATCHUP'].str.split().str[-1],
                'is_home': ~df['MATCHUP'].str.contains('@'),
                'points': df['PTS'].astype(float),
                'rebounds': df['REB'].astype(float),
                'assists': df['AST'].astype(float),
                'minutes': df['MIN'].astype(float),
                'fg_pct': df['FG_PCT'].astype(float) * 100,
                'result': df['WL']
            })
            
            # Ajoute variables pour régression
            df_clean['opponent_def_rating'] = df_clean['opponent'].apply(
                lambda x: self.defensive_ratings.get(x, 112.0)
            )
            df_clean['rest_days'] = 1
            df_clean['back_to_back'] = 0
            df_clean['team_pace'] = 100.0
            
            self.cache[cache_key] = df_clean
            
            print(f"✅ {len(df_clean)} matchs réels récupérés")
            print(f"   Moyenne: {df_clean['points'].mean():.1f} pts")
            
            return df_clean.head(n_games)
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return None
    
    def get_simulated_data(self, player_name, n_games=20):
        """
        Données simulées si nba_api non disponible
        """
        np.random.seed(hash(player_name) % 2**32)
        base_points = np.random.uniform(20, 28)
        
        games = []
        for i in range(n_games):
            game = {
                'date': (datetime.now()).strftime('%Y-%m-%d'),
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
    
    def get_player_data(self, player_name, n_games=20):
        """
        Récupère les données (vraies ou simulées)
        """
        if NBA_API_AVAILABLE:
            df = self.get_real_player_data(player_name, n_games)
            if df is not None:
                return df, True  # True = vraies données
        
        # Fallback sur simulé
        print(f"⚠️ Utilise données simulées pour {player_name}")
        return self.get_simulated_data(player_name, n_games), False
    
    def build_model(self, player_name):
        """
        Construit le modèle de régression
        """
        df, is_real = self.get_player_data(player_name, n_games=20)
        
        # Variables indépendantes
        X = df[['is_home', 'opponent_def_rating', 'minutes', 'rest_days', 'back_to_back', 'team_pace']]
        y = df['points']
        
        # Régression
        model = LinearRegression()
        model.fit(X, y)
        
        # Prédictions
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Statistiques
        n = len(y)
        k = X.shape[1]
        dof = n - k - 1
        
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        
        # P-values
        var_residuals = np.sum(residuals**2) / dof
        var_coef = var_residuals * np.linalg.inv(X.T @ X).diagonal()
        std_errors = np.sqrt(var_coef)
        t_stats = model.coef_ / std_errors
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
        
        # Intervalles de confiance
        t_critical = stats.t.ppf(0.975, dof)
        conf_intervals = [
            (coef - t_critical * se, coef + t_critical * se)
            for coef, se in zip(model.coef_, std_errors)
        ]
        
        regression_stats = {
            'coefficients': [
                {
                    'variable': var,
                    'coefficient': float(coef),
                    'std_error': float(se),
                    't_stat': float(t),
                    'p_value': float(p),
                    'significant': p < 0.05
                }
                for var, coef, se, t, p in zip(
                    ['is_home', 'opponent_def_rating', 'minutes', 'rest_days', 'back_to_back', 'team_pace'],
                    model.coef_, std_errors, t_stats, p_values
                )
            ],
            'model_quality': {
                'r_squared': float(r2),
                'adjusted_r_squared': float(1 - (1 - r2) * (n - 1) / dof),
                'rmse': float(rmse),
                'sample_size': int(n)
            }
        }
        
        return model, regression_stats, df, is_real
    
    def predict(self, player_name, opponent='LAL', is_home=True, line=None):
        """
        Fait une prédiction complète
        """
        # Construit le modèle
        model, reg_stats, df, is_real = self.build_model(player_name)
        
        # Prépare la prédiction
        opp_def = self.defensive_ratings.get(opponent, 112.0)
        X_pred = np.array([[
            1 if is_home else 0,
            opp_def,
            35,  # minutes
            1,   # rest_days
            0,   # back_to_back
            100  # pace
        ]])
        
        # Prédiction
        predicted_points = float(model.predict(X_pred)[0])
        std_dev = float(df['points'].std())
        
        # Intervalle de confiance 80%
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
        implied_prob = 0.5  # -110 odds
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
            'data_source': '✅ VRAIES DONNÉES (nba_api)' if is_real else '⚠️ Données simulées',
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
            'regression_stats': reg_stats,
            'recent_games': df[['date', 'opponent', 'points', 'result']].head(5).to_dict('records')
        }
    
    def get_top_opportunities(self, min_edge=5.0):
        """
        Trouve les meilleures opportunités
        """
        top_players = [
            'LeBron James', 'Stephen Curry', 'Luka Doncic',
            'Kevin Durant', 'Giannis Antetokounmpo', 'Nikola Jokic',
            'Joel Embiid', 'Jayson Tatum'
        ]
        
        opportunities = []
        for player in top_players:
            try:
                result = self.predict(player, 'GSW', True)
                if result['recommendation']['edge'] >= min_edge:
                    opportunities.append(result)
            except Exception as e:
                print(f"Erreur pour {player}: {e}")
                continue
        
        # Trie par edge
        opportunities.sort(key=lambda x: x['recommendation']['edge'], reverse=True)
        return opportunities

# Initialize analyzer
analyzer = CompleteNBAAnalyzer()

# API Routes
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

@app.route('/api/opportunities', methods=['GET'])
def opportunities():
    """Récupère les meilleures opportunités"""
    min_edge = request.args.get('min_edge', 5.0, type=float)
    opps = analyzer.get_top_opportunities(min_edge)
    return jsonify(opps)

@app.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({
        'status': 'ok',
        'nba_api': NBA_API_AVAILABLE,
        'data_source': 'REAL' if NBA_API_AVAILABLE else 'SIMULATED',
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print("\n🌐 Server démarré sur: http://localhost:5000")
    print("\n📚 Endpoints disponibles:")
    print("   GET  /api/health - Status du système")
    print("   GET  /api/opportunities?min_edge=5.0 - Meilleures opportunités")
    print("   POST /api/analyze - Analyse un joueur")
    print("\n🎯 Exemple:")
    print('   curl -X POST http://localhost:5000/api/analyze \\')
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"player": "LeBron James", "opponent": "GSW", "is_home": true}\'')
    print("\n" + "=" * 70)
    print("✅ Ouvre advanced_interface.html dans ton navigateur")
    print("=" * 70 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
