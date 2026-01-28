"""
NBA Betting Analyzer - Real Data Version v2
Uses nba_api with weighted temporal analysis and detailed splits
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

app = Flask(__name__)
CORS(app)

# Import nba_api
try:
    from nba_api.stats.endpoints import playergamelog, commonplayerinfo
    from nba_api.stats.static import players, teams
    NBA_API_AVAILABLE = True
    print("✅ nba_api imported successfully")
except ImportError as e:
    NBA_API_AVAILABLE = False
    print(f"❌ nba_api not available: {e}")

class NBAAnalyzer:
    """Analyste NBA avec vraies données, pondération temporelle et splits"""
    
    def __init__(self):
        self.current_season = '2024-25'
        
        # Ratings défensifs moyens par équipe
        self.defensive_ratings = {
            'ATL': 115.2, 'BOS': 110.5, 'BKN': 114.8, 'CHA': 116.1, 'CHI': 113.9,
            'CLE': 108.2, 'DAL': 112.7, 'DEN': 111.3, 'DET': 115.8, 'GSW': 112.1,
            'HOU': 110.9, 'IND': 116.5, 'LAC': 111.8, 'LAL': 113.4, 'MEM': 112.6,
            'MIA': 111.2, 'MIL': 112.4, 'MIN': 108.9, 'NOP': 114.3, 'NYK': 110.7,
            'OKC': 109.1, 'ORL': 108.6, 'PHI': 113.1, 'PHX': 114.2, 'POR': 115.7,
            'SAC': 114.9, 'SAS': 116.0, 'TOR': 115.4, 'UTA': 115.3, 'WAS': 116.8
        }
    
    def get_player_id(self, player_name):
        """Trouve l'ID d'un joueur par son nom"""
        if not NBA_API_AVAILABLE:
            return None
        
        try:
            all_players = players.get_players()
            player = [p for p in all_players if player_name.lower() in p['full_name'].lower()]
            
            if player:
                return player[0]['id']
            return None
        except Exception as e:
            print(f"Error finding player: {e}")
            return None
    
    def get_season_games(self, player_id):
        """Récupère TOUS les matchs de la saison en cours"""
        if not NBA_API_AVAILABLE or not player_id:
            return None
        
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id,
                season=self.current_season
            )
            df = gamelog.get_data_frames()[0]
            
            if df.empty:
                return None
            
            # Convertit les colonnes importantes
            df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')
            df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # Détermine home/away (vs. = home, @ = away)
            df['IS_HOME'] = df['MATCHUP'].str.contains('vs.')
            
            # Extrait l'adversaire (3 lettres après vs. ou @)
            df['OPPONENT'] = df['MATCHUP'].str.extract(r'(?:vs\.|@)\s*([A-Z]{3})')[0]
            
            # Trie par date (plus récent en premier)
            df = df.sort_values('GAME_DATE', ascending=False).reset_index(drop=True)
            
            return df
            
        except Exception as e:
            print(f"Error fetching games: {e}")
            return None
    
    def calculate_weighted_average(self, games_df):
        """
        Calcule moyenne pondérée selon récence
        - Derniers 10 matchs: 50%
        - Matchs 11-30: 30%
        - Reste: 20%
        """
        if games_df is None or len(games_df) == 0:
            return 0, 0
        
        points = games_df['PTS'].values
        n = len(points)
        
        # Crée poids
        weights = np.zeros(n)
        
        # Derniers 10: poids 0.5
        if n >= 10:
            weights[:10] = 0.5 / 10
        else:
            weights[:n] = 0.5 / n
        
        # Matchs 11-30: poids 0.3
        if n > 10:
            end_idx = min(30, n)
            count = end_idx - 10
            weights[10:end_idx] = 0.3 / count
        
        # Reste: poids 0.2
        if n > 30:
            count = n - 30
            weights[30:] = 0.2 / count
        
        # Normalise les poids (au cas où)
        weights = weights / weights.sum()
        
        # Moyenne pondérée
        weighted_avg = np.average(points, weights=weights)
        
        # Écart-type pondéré
        weighted_std = np.sqrt(np.average((points - weighted_avg)**2, weights=weights))
        
        return weighted_avg, weighted_std
    
    def calculate_splits(self, games_df, opponent=None):
        """Calcule statistiques détaillées (home/away, vs opponent)"""
        if games_df is None or len(games_df) == 0:
            return {}
        
        splits = {}
        
        # Home games
        home_games = games_df[games_df['IS_HOME'] == True]
        if len(home_games) > 0:
            splits['home'] = {
                'avg': round(home_games['PTS'].mean(), 1),
                'std': round(home_games['PTS'].std(), 1),
                'games': len(home_games)
            }
        
        # Away games
        away_games = games_df[games_df['IS_HOME'] == False]
        if len(away_games) > 0:
            splits['away'] = {
                'avg': round(away_games['PTS'].mean(), 1),
                'std': round(away_games['PTS'].std(), 1),
                'games': len(away_games)
            }
        
        # vs specific opponent
        if opponent:
            vs_opp = games_df[games_df['OPPONENT'] == opponent]
            if len(vs_opp) > 0:
                splits['vs_opponent'] = {
                    'avg': round(vs_opp['PTS'].mean(), 1),
                    'std': round(vs_opp['PTS'].std(), 1),
                    'games': len(vs_opp),
                    'last_3': vs_opp.head(3)['PTS'].tolist() if len(vs_opp) >= 3 else vs_opp['PTS'].tolist()
                }
        
        return splits
    
    def calculate_trend(self, games_df, num_games=10):
        """Calcule la tendance récente (régression linéaire)"""
        if games_df is None or len(games_df) < 5:
            return 0, 0
        
        try:
            recent = games_df.head(num_games)
            X = np.arange(len(recent)).reshape(-1, 1)
            y = recent['PTS'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            trend = model.coef_[0]
            r2 = r2_score(y, model.predict(X))
            
            return trend, r2
            
        except Exception as e:
            print(f"Error calculating trend: {e}")
            return 0, 0
    
    def adjust_for_matchup(self, base_prediction, opponent, is_home, splits):
        """
        Ajuste prédiction selon:
        1. Défense adverse
        2. Home/Away performance réelle du joueur
        3. Performance historique vs cet adversaire
        """
        adjusted = base_prediction
        adjustments = {}
        
        # 1. Défense adverse
        avg_rating = 113.0
        opp_rating = self.defensive_ratings.get(opponent, avg_rating)
        defense_factor = opp_rating / avg_rating
        adjusted *= defense_factor
        adjustments['defense'] = f"{((defense_factor - 1) * 100):.1f}%"
        
        # 2. Home/Away split du joueur
        if is_home and 'home' in splits:
            # Compare home avg vs overall
            home_diff = splits['home']['avg'] - base_prediction
            adjusted += home_diff * 0.5  # 50% de l'écart
            adjustments['home_away'] = f"+{home_diff * 0.5:.1f} pts (home)"
        elif not is_home and 'away' in splits:
            away_diff = splits['away']['avg'] - base_prediction
            adjusted += away_diff * 0.5
            adjustments['home_away'] = f"{away_diff * 0.5:+.1f} pts (away)"
        
        # 3. vs Opponent historique
        if 'vs_opponent' in splits and splits['vs_opponent']['games'] >= 3:
            opp_avg = splits['vs_opponent']['avg']
            opp_diff = opp_avg - base_prediction
            # Plus de poids si beaucoup de matchs
            weight = min(splits['vs_opponent']['games'] / 10, 0.3)
            adjusted += opp_diff * weight
            adjustments['vs_opponent'] = f"{opp_diff * weight:+.1f} pts (history)"
        
        return adjusted, adjustments
    
    def predict_points(self, player_name, opponent, is_home=True, line=None):
        """
        Prédit les points avec analyse complète
        """
        if not NBA_API_AVAILABLE:
            return {
                'error': 'nba_api not available',
                'player': player_name,
                'status': 'API_UNAVAILABLE'
            }
        
        # 1. Trouve le joueur
        player_id = self.get_player_id(player_name)
        if not player_id:
            return {
                'error': f'Player not found: {player_name}',
                'status': 'PLAYER_NOT_FOUND'
            }
        
        # 2. Récupère TOUTE la saison
        season_games = self.get_season_games(player_id)
        if season_games is None or len(season_games) < 10:
            return {
                'error': 'Not enough games this season',
                'player': player_name,
                'status': 'INSUFFICIENT_DATA'
            }
        
        # 3. Calcule moyenne pondérée
        weighted_avg, weighted_std = self.calculate_weighted_average(season_games)
        
        # 4. Calcule splits
        splits = self.calculate_splits(season_games, opponent)
        
        # 5. Calcule tendance récente
        trend, trend_quality = self.calculate_trend(season_games, num_games=10)
        
        # 6. Prédiction de base (moyenne pondérée + tendance)
        base_prediction = weighted_avg + (trend * 1.5)
        
        # 7. Ajuste selon matchup
        final_prediction, adjustment_details = self.adjust_for_matchup(
            base_prediction, opponent, is_home, splits
        )
        
        # 8. Intervalle de confiance (95%)
        n = len(season_games)
        se = weighted_std / np.sqrt(n)
        confidence_interval = stats.t.interval(
            0.95,
            n - 1,
            loc=final_prediction,
            scale=se
        )
        
        # 9. Analyse ligne bookmaker
        recommendation = None
        over_probability = None
        edge = None
        
        if line is not None:
            z_score = (line - final_prediction) / weighted_std
            over_probability = 1 - stats.norm.cdf(z_score)
            edge = over_probability - 0.5
            
            # Recommandation
            if over_probability >= 0.58 and edge >= 0.08:
                recommendation = 'OVER'
            elif over_probability <= 0.42 and edge <= -0.08:
                recommendation = 'UNDER'
            else:
                recommendation = 'SKIP'
        
        # 10. Résultats
        result = {
            'player': player_name,
            'opponent': opponent,
            'is_home': is_home,
            'prediction': round(final_prediction, 1),
            'confidence_interval': {
                'lower': round(confidence_interval[0], 1),
                'upper': round(confidence_interval[1], 1),
                'confidence_level': '95%'
            },
            'season_stats': {
                'weighted_avg': round(weighted_avg, 1),
                'std_dev': round(weighted_std, 1),
                'games_played': len(season_games),
                'trend': round(trend, 2),
                'trend_quality': round(trend_quality, 2)
            },
            'splits': splits,
            'adjustments': adjustment_details,
            'timestamp': datetime.now().isoformat()
        }
        
        if line is not None:
            result['line_analysis'] = {
                'bookmaker_line': line,
                'over_probability': round(over_probability, 3),
                'under_probability': round(1 - over_probability, 3),
                'edge': round(edge, 3),
                'recommendation': recommendation,
                'confidence': 'HIGH' if abs(edge) >= 0.12 else 'MEDIUM' if abs(edge) >= 0.08 else 'LOW',
                'kelly_criterion': round(edge * 2, 3) if abs(edge) >= 0.08 else 0
            }
        
        result['status'] = 'SUCCESS'
        return result


# Initialize analyzer
analyzer = NBAAnalyzer()

@app.route('/', methods=['GET'])
def root():
    """Route racine - redirige vers /api"""
    return jsonify({
        'message': 'NBA Betting Analyzer API',
        'status': 'online',
        'endpoints': {
            'info': '/api',
            'health': '/api/health',
            'analyze': 'POST /api/analyze'
        }
    })

@app.route('/api', methods=['GET'])
def api_info():
    return jsonify({
        'name': 'NBA Betting Analyzer API',
        'version': '2.1',
        'data_source': 'nba_api (REAL DATA - Full Season)',
        'nba_api_status': 'AVAILABLE' if NBA_API_AVAILABLE else 'UNAVAILABLE',
        'features': [
            'Weighted temporal analysis (50% recent, 30% mid, 20% old)',
            'Home/Away splits',
            'Head-to-head history',
            'Opponent defense adjustment',
            '95% confidence intervals',
            'Kelly Criterion bet sizing'
        ],
        'endpoints': {
            'health': 'GET /api/health',
            'analyze': 'POST /api/analyze'
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'nba_api': NBA_API_AVAILABLE,
        'data_source': 'REAL (Full Season)' if NBA_API_AVAILABLE else 'UNAVAILABLE',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_player():
    """
    Analyse complète d'un joueur
    
    Body:
    {
        "player": "LeBron James",
        "opponent": "GSW",
        "is_home": true,
        "line": 25.5
    }
    """
    try:
        data = request.json
        
        player = data.get('player')
        opponent = data.get('opponent')
        is_home = data.get('is_home', True)
        line = data.get('line')
        
        if not player or not opponent:
            return jsonify({
                'error': 'Missing required fields: player, opponent'
            }), 400
        
        result = analyzer.predict_points(player, opponent, is_home, line)
        
        if result.get('status') == 'SUCCESS':
            return jsonify(result)
        else:
            return jsonify(result), 404
            
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'ERROR'
        }), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
