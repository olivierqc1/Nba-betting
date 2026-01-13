from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from scipy import stats
import time
from betonline_scraper import BetOnlineScraper

app = Flask(__name__)
CORS(app)

# Initialiser le scraper BetOnline
betonline = BetOnlineScraper()

# Mapping des équipes NBA
NBA_TEAMS = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'BRK': 'Brooklyn Nets',
    'CHO': 'Charlotte Hornets', 'CHI': 'Chicago Bulls', 'CLE': 'Cleveland Cavaliers',
    'DAL': 'Dallas Mavericks', 'DEN': 'Denver Nuggets', 'DET': 'Detroit Pistons',
    'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets', 'IND': 'Indiana Pacers',
    'LAC': 'LA Clippers', 'LAL': 'Los Angeles Lakers', 'MEM': 'Memphis Grizzlies',
    'MIA': 'Miami Heat', 'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves',
    'NOP': 'New Orleans Pelicans', 'NYK': 'New York Knicks', 'OKC': 'Oklahoma City Thunder',
    'ORL': 'Orlando Magic', 'PHI': 'Philadelphia 76ers', 'PHO': 'Phoenix Suns',
    'POR': 'Portland Trail Blazers', 'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'WAS': 'Washington Wizards'
}

def get_player_last_games(player_name, player_id, num_games=10):
    """Récupère les N derniers matchs d'un joueur avec poids décroissant"""
    try:
        url = f"https://www.basketball-reference.com/players/{player_id[0]}/{player_id}/gamelog/2025"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table', {'id': 'pgl_basic'})
        if not table:
            return None
            
        rows = table.find('tbody').find_all('tr')
        games = []
        
        for row in rows[:num_games]:
            if 'class' in row.attrs and 'thead' in row.attrs['class']:
                continue
                
            try:
                game_data = {
                    'date': row.find('td', {'data-stat': 'date_game'}).text,
                    'opponent': row.find('td', {'data-stat': 'opp_id'}).text,
                    'home_away': row.find('td', {'data-stat': 'game_location'}).text,
                    'points': float(row.find('td', {'data-stat': 'pts'}).text or 0),
                    'rebounds': float(row.find('td', {'data-stat': 'trb'}).text or 0),
                    'assists': float(row.find('td', {'data-stat': 'ast'}).text or 0),
                    'minutes': float(row.find('td', {'data-stat': 'mp'}).text.split(':')[0] or 0)
                }
                games.append(game_data)
            except:
                continue
        
        return games[:num_games]
    except Exception as e:
        print(f"Erreur lors du scraping: {e}")
        return None

def calculate_weighted_prediction(games, stat_type='points'):
    """
    Calcule la prédiction avec poids exponentiel décroissant AMÉLIORÉ
    Plus le match est récent, plus le poids est important
    Decay factor de 0.75 pour donner encore PLUS de poids aux matchs récents
    """
    if not games or len(games) == 0:
        return None, None
    
    # Créer des poids exponentiels avec decay factor plus agressif
    # Les 3 derniers matchs comptent pour ~60% du poids total
    n = len(games)
    weights = np.array([0.75 ** i for i in range(n)])  # Changé de 0.85 à 0.75
    weights = weights / weights.sum()  # Normaliser pour que la somme = 1
    
    # Extraire les valeurs de la stat
    values = np.array([g[stat_type] for g in games])
    
    # Calculer la moyenne pondérée
    weighted_mean = np.sum(values * weights)
    
    # Calculer l'écart-type pondéré
    weighted_variance = np.sum(weights * (values - weighted_mean) ** 2)
    weighted_std = np.sqrt(weighted_variance)
    
    # Intervalle de confiance à 95% (1.96 * std pour distribution normale)
    ci_margin = 1.96 * weighted_std
    ci_lower = weighted_mean - ci_margin
    ci_upper = weighted_mean + ci_margin
    
    return {
        'prediction': round(weighted_mean, 1),
        'std': round(weighted_std, 2),
        'ci_lower': round(max(0, ci_lower), 1),  # Ne peut pas être négatif
        'ci_upper': round(ci_upper, 1),
        'recent_values': values[:5].tolist(),  # 5 dernières valeurs pour graphique
        'weights': weights[:5].tolist()
    }

def get_opponent_defense_rating(opponent_abbr):
    """Récupère le rating défensif de l'équipe adverse (simplifié pour démo)"""
    # Dans une version complète, on scraperait les vraies stats défensives
    # Pour la démo, on utilise des valeurs estimées
    defense_ratings = {
        'BOS': 108.5, 'MIL': 109.2, 'PHI': 110.8, 'MIA': 111.5,
        'CLE': 109.8, 'NYK': 112.3, 'BRK': 115.6, 'ATL': 114.2,
        'CHO': 116.8, 'CHI': 113.5, 'WAS': 118.2, 'ORL': 112.8,
        'IND': 114.5, 'DET': 117.3, 'TOR': 115.9, 'GSW': 111.2,
        'LAL': 112.7, 'PHO': 113.8, 'SAC': 115.1, 'LAC': 111.8,
        'DEN': 110.5, 'MIN': 109.9, 'OKC': 108.8, 'POR': 116.5,
        'UTA': 114.8, 'DAL': 112.4, 'MEM': 113.2, 'NOP': 115.7,
        'SAS': 117.8, 'HOU': 114.9
    }
    return defense_ratings.get(opponent_abbr, 113.0)  # Moyenne de la ligue

def adjust_for_opponent(prediction, opponent_abbr):
    """Ajuste la prédiction selon la qualité défensive de l'adversaire"""
    opp_rating = get_opponent_defense_rating(opponent_abbr)
    league_avg = 113.0
    
    # Si défense faible (rating élevé), augmenter la prédiction
    # Si défense forte (rating bas), diminuer la prédiction
    adjustment_factor = 1 + ((opp_rating - league_avg) / league_avg) * 0.3
    
    adjusted_pred = prediction * adjustment_factor
    return round(adjusted_pred, 1)

def calculate_roi(prediction, ci_lower, ci_upper, line, odds):
    """
    Calcule le ROI attendu d'un pari
    
    prediction: valeur prédite
    ci_lower/ci_upper: intervalle de confiance à 95%
    line: ligne de pari (ex: 25.5 points)
    odds: cote américaine (ex: -110)
    """
    # Convertir les cotes américaines en probabilité implicite
    if odds < 0:
        implied_prob = abs(odds) / (abs(odds) + 100)
    else:
        implied_prob = 100 / (odds + 100)
    
    # Calculer notre probabilité estimée que le pari passe
    # On utilise la distance entre la prédiction et la ligne, normalisée par l'écart-type
    std = (ci_upper - ci_lower) / (2 * 1.96)
    z_score = (prediction - line) / std if std > 0 else 0
    
    # Probabilité que le joueur dépasse la ligne (distribution normale)
    our_prob = stats.norm.cdf(z_score)
    
    # Si on parie "under", inverser la probabilité
    # Pour l'instant on assume "over"
    
    # Calcul du ROI attendu
    # ROI = (probabilité de gagner * gain) - (probabilité de perdre * mise)
    if odds < 0:
        potential_win = 100 / abs(odds)  # Pour 1$ misé
    else:
        potential_win = odds / 100
    
    expected_value = (our_prob * potential_win) - ((1 - our_prob) * 1)
    roi_percent = expected_value * 100
    
    # Calculer l'edge (notre prob - leur prob)
    edge = (our_prob - implied_prob) * 100
    
    return {
        'roi': round(roi_percent, 1),
        'our_probability': round(our_prob * 100, 1),
        'implied_probability': round(implied_prob * 100, 1),
        'edge': round(edge, 1),
        'confidence': 'high' if abs(z_score) > 1.5 else 'medium' if abs(z_score) > 0.8 else 'low'
    }

@app.route('/api/players', methods=['GET'])
def get_players():
    """Retourne la liste des joueurs disponibles sur BetOnline avec leurs props"""
    try:
        # Récupérer les props depuis BetOnline
        betonline_props = betonline.get_nba_player_props()
        
        # Mapping des IDs Basketball-Reference pour les joueurs populaires
        player_ids = {
            'LeBron James': 'jamesle01',
            'Stephen Curry': 'curryst01',
            'Giannis Antetokounmpo': 'antetgi01',
            'Kevin Durant': 'duranke01',
            'Luka Doncic': 'doncilu01',
            'Nikola Jokic': 'jokicni01',
            'Joel Embiid': 'embiijo01',
            'Damian Lillard': 'lillada01',
            'Jayson Tatum': 'tatumja01',
            'Anthony Davis': 'davisan02',
            'Kawhi Leonard': 'leonaka01',
            'Jimmy Butler': 'butleji01',
            'Devin Booker': 'bookede01',
            'Trae Young': 'youngtr01',
            'Donovan Mitchell': 'mitchdo01'
        }
        
        # Enrichir les données avec les IDs BBRef
        for player_data in betonline_props:
            player_name = player_data['player']
            if player_name in player_ids:
                player_data['id'] = player_ids[player_name]
                player_data['team'] = self._extract_team_from_matchup(player_data['matchup'])
            else:
                # Essayer de deviner l'ID pour les autres joueurs
                player_data['id'] = self._generate_player_id(player_name)
                player_data['team'] = 'N/A'
        
        return jsonify(betonline_props)
    except Exception as e:
        print(f"Erreur lors de la récupération des joueurs: {e}")
        # Fallback sur l'ancienne méthode si BetOnline échoue
        popular_players = [
            {'name': 'LeBron James', 'id': 'jamesle01', 'team': 'LAL'},
            {'name': 'Stephen Curry', 'id': 'curryst01', 'team': 'GSW'},
            {'name': 'Giannis Antetokounmpo', 'id': 'antetgi01', 'team': 'MIL'},
            {'name': 'Kevin Durant', 'id': 'duranke01', 'team': 'PHO'},
            {'name': 'Luka Doncic', 'id': 'doncilu01', 'team': 'DAL'},
            {'name': 'Nikola Jokic', 'id': 'jokicni01', 'team': 'DEN'},
            {'name': 'Joel Embiid', 'id': 'embiijo01', 'team': 'PHI'},
            {'name': 'Damian Lillard', 'id': 'lillada01', 'team': 'MIL'},
            {'name': 'Jayson Tatum', 'id': 'tatumja01', 'team': 'BOS'},
            {'name': 'Anthony Davis', 'id': 'davisan02', 'team': 'LAL'},
        ]
        return jsonify(popular_players)

def _extract_team_from_matchup(matchup):
    """Extrait l'équipe depuis le format 'LAL vs GSW' ou 'GSW @ LAL'"""
    teams = re.findall(r'([A-Z]{2,3})', matchup)
    return teams[0] if teams else 'N/A'

def _generate_player_id(player_name):
    """Génère un ID BBRef basique pour un joueur"""
    parts = player_name.lower().split()
    if len(parts) >= 2:
        last = parts[-1]
        first = parts[0]
        return f"{last[:5]}{first[:2]}01"
    return "unknown01"

@app.route('/api/analyze', methods=['POST'])
def analyze_player():
    """Analyse un joueur avec les VRAIES lignes de BetOnline et retourne seulement les paris avec ROI ≥ 80%"""
    data = request.json
    player_name = data.get('player_name')
    player_id = data.get('player_id')
    opponent = data.get('opponent', 'BOS')
    is_home = data.get('is_home', True)
    
    # Récupérer les 10 derniers matchs
    games = get_player_last_games(player_name, player_id, num_games=10)
    
    if not games:
        return jsonify({'error': 'Impossible de récupérer les données du joueur'}), 400
    
    # Récupérer les props disponibles sur BetOnline pour ce joueur
    betonline_player_data = betonline.get_player_by_name(player_name)
    
    recommendations = []
    
    if betonline_player_data and betonline_player_data.get('props'):
        # Utiliser les VRAIES lignes de BetOnline
        for prop in betonline_player_data['props']:
            stat = prop['stat_type']
            line = prop['line']
            over_odds = prop['over_odds']
            under_odds = prop['under_odds']
            
            # Calculer la prédiction pour cette stat
            prediction_data = calculate_weighted_prediction(games, stat)
            if not prediction_data:
                continue
            
            # Ajuster selon l'adversaire
            adjusted_pred = adjust_for_opponent(prediction_data['prediction'], opponent)
            
            # Calculer le ROI pour OVER
            roi_over = calculate_roi(
                adjusted_pred,
                prediction_data['ci_lower'],
                prediction_data['ci_upper'],
                line,
                over_odds
            )
            
            # Calculer le ROI pour UNDER (inverser la probabilité)
            roi_under = calculate_roi_under(
                adjusted_pred,
                prediction_data['ci_lower'],
                prediction_data['ci_upper'],
                line,
                under_odds
            )
            
            # Ajouter les recommandations avec ROI ≥ 80%
            stat_french = {
                'points': 'Points',
                'rebounds': 'Rebonds',
                'assists': 'Passes',
                'threes': '3-Points',
                'steals': 'Interceptions',
                'blocks': 'Contres'
            }.get(stat, stat.capitalize())
            
            if roi_over['roi'] >= 80:
                recommendations.append({
                    'player': player_name,
                    'stat': stat,
                    'stat_french': stat_french,
                    'prediction': adjusted_pred,
                    'ci_lower': prediction_data['ci_lower'],
                    'ci_upper': prediction_data['ci_upper'],
                    'line': line,
                    'bet_type': 'OVER',
                    'odds': over_odds,
                    'roi': roi_over['roi'],
                    'our_probability': roi_over['our_probability'],
                    'implied_probability': roi_over['implied_probability'],
                    'edge': roi_over['edge'],
                    'confidence': roi_over['confidence'],
                    'recent_values': prediction_data['recent_values'],
                    'opponent': opponent,
                    'is_home': is_home,
                    'defense_rating': get_opponent_defense_rating(opponent),
                    'matchup': betonline_player_data.get('matchup', ''),
                    'source': betonline_player_data.get('source', 'BetOnline')
                })
            
            if roi_under['roi'] >= 80:
                recommendations.append({
                    'player': player_name,
                    'stat': stat,
                    'stat_french': stat_french,
                    'prediction': adjusted_pred,
                    'ci_lower': prediction_data['ci_lower'],
                    'ci_upper': prediction_data['ci_upper'],
                    'line': line,
                    'bet_type': 'UNDER',
                    'odds': under_odds,
                    'roi': roi_under['roi'],
                    'our_probability': roi_under['our_probability'],
                    'implied_probability': roi_under['implied_probability'],
                    'edge': roi_under['edge'],
                    'confidence': roi_under['confidence'],
                    'recent_values': prediction_data['recent_values'],
                    'opponent': opponent,
                    'is_home': is_home,
                    'defense_rating': get_opponent_defense_rating(opponent),
                    'matchup': betonline_player_data.get('matchup', ''),
                    'source': betonline_player_data.get('source', 'BetOnline')
                })
    
    else:
        # Fallback: simuler des lignes si BetOnline n'a pas de données pour ce joueur
        for stat in ['points', 'rebounds', 'assists']:
            prediction_data = calculate_weighted_prediction(games, stat)
            if not prediction_data:
                continue
            
            adjusted_pred = adjust_for_opponent(prediction_data['prediction'], opponent)
            
            # Créer plusieurs lignes autour de la prédiction
            lines_to_check = [
                adjusted_pred - 2.5,
                adjusted_pred - 1.5,
                adjusted_pred - 0.5,
                adjusted_pred + 0.5,
                adjusted_pred + 1.5,
                adjusted_pred + 2.5
            ]
            
            for line in lines_to_check:
                odds = -110
                roi_data = calculate_roi(
                    adjusted_pred,
                    prediction_data['ci_lower'],
                    prediction_data['ci_upper'],
                    line,
                    odds
                )
                
                if roi_data['roi'] >= 80:
                    recommendations.append({
                        'player': player_name,
                        'stat': stat,
                        'stat_french': {'points': 'Points', 'rebounds': 'Rebonds', 'assists': 'Passes'}[stat],
                        'prediction': adjusted_pred,
                        'ci_lower': prediction_data['ci_lower'],
                        'ci_upper': prediction_data['ci_upper'],
                        'line': round(line, 1),
                        'bet_type': 'OVER' if adjusted_pred > line else 'UNDER',
                        'odds': odds,
                        'roi': roi_data['roi'],
                        'our_probability': roi_data['our_probability'],
                        'implied_probability': roi_data['implied_probability'],
                        'edge': roi_data['edge'],
                        'confidence': roi_data['confidence'],
                        'recent_values': prediction_data['recent_values'],
                        'opponent': opponent,
                        'is_home': is_home,
                        'defense_rating': get_opponent_defense_rating(opponent),
                        'source': 'Simulé'
                    })
    
    # Trier par ROI décroissant
    recommendations.sort(key=lambda x: x['roi'], reverse=True)
    
    return jsonify({
        'player': player_name,
        'games_analyzed': len(games),
        'recommendations': recommendations[:10],  # Top 10 recommandations
        'total_opportunities': len(recommendations)
    })

def calculate_roi_under(prediction, ci_lower, ci_upper, line, odds):
    """
    Calcule le ROI pour un pari UNDER
    """
    # Convertir les cotes américaines en probabilité implicite
    if odds < 0:
        implied_prob = abs(odds) / (abs(odds) + 100)
    else:
        implied_prob = 100 / (odds + 100)
    
    # Calculer notre probabilité estimée que le pari UNDER passe
    std = (ci_upper - ci_lower) / (2 * 1.96)
    z_score = (prediction - line) / std if std > 0 else 0
    
    # Probabilité que le joueur soit EN DESSOUS de la ligne
    our_prob = 1 - stats.norm.cdf(z_score)
    
    # Calcul du ROI attendu
    if odds < 0:
        potential_win = 100 / abs(odds)
    else:
        potential_win = odds / 100
    
    expected_value = (our_prob * potential_win) - ((1 - our_prob) * 1)
    roi_percent = expected_value * 100
    
    # Calculer l'edge
    edge = (our_prob - implied_prob) * 100
    
    return {
        'roi': round(roi_percent, 1),
        'our_probability': round(our_prob * 100, 1),
        'implied_probability': round(implied_prob * 100, 1),
        'edge': round(edge, 1),
        'confidence': 'high' if abs(z_score) > 1.5 else 'medium' if abs(z_score) > 0.8 else 'low'
    }

if __name__ == '__main__':
    app.run(debug=True, port=5000)
