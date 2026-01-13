#!/usr/bin/env python3
"""
BetOnline Scraper - Récupère les lignes de paris NBA en temps réel
"""

import requests
from bs4 import BeautifulSoup
import json
from datetime import datetime
import re

class BetOnlineScraper:
    def __init__(self):
        self.base_url = "https://www.betonline.ag"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def get_nba_player_props(self):
        """
        Récupère les props de joueurs NBA (points, rebounds, assists)
        """
        try:
            # URL pour les props NBA player sur BetOnline
            url = f"{self.base_url}/sportsbook/basketball/nba"
            response = requests.get(url, headers=self.headers, timeout=10)
            
            if response.status_code != 200:
                print(f"Erreur: Status code {response.status_code}")
                return self._get_fallback_data()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse les props - structure spécifique à BetOnline
            props = self._parse_betonline_props(soup)
            
            if not props:
                print("Aucune prop trouvée, utilisation des données de secours")
                return self._get_fallback_data()
            
            return props
            
        except Exception as e:
            print(f"Erreur scraping BetOnline: {e}")
            return self._get_fallback_data()
    
    def _parse_betonline_props(self, soup):
        """
        Parse la structure HTML de BetOnline pour extraire les props
        """
        props = []
        
        # Cherche les sections de props de joueurs
        prop_sections = soup.find_all('div', class_=re.compile('player.*prop', re.I))
        
        for section in prop_sections:
            try:
                # Extraire nom du joueur
                player_name = section.find('span', class_='player-name')
                if not player_name:
                    continue
                
                player = player_name.text.strip()
                
                # Extraire la ligne (over/under)
                line_elem = section.find('span', class_='line')
                if not line_elem:
                    continue
                    
                line = float(line_elem.text.strip())
                
                # Extraire les odds
                over_odds_elem = section.find('span', class_='over-odds')
                under_odds_elem = section.find('span', class_='under-odds')
                
                over_odds = over_odds_elem.text.strip() if over_odds_elem else "-110"
                under_odds = under_odds_elem.text.strip() if under_odds_elem else "-110"
                
                # Extraire le type de prop (points, rebounds, assists)
                prop_type_elem = section.find('span', class_='prop-type')
                prop_type = prop_type_elem.text.strip() if prop_type_elem else "Points"
                
                # Extraire l'adversaire et domicile/extérieur
                matchup_elem = section.find('span', class_='matchup')
                opponent = "Unknown"
                is_home = True
                
                if matchup_elem:
                    matchup_text = matchup_elem.text.strip()
                    if '@' in matchup_text:
                        is_home = False
                        opponent = matchup_text.split('@')[1].strip()
                    else:
                        opponent = matchup_text.split('vs')[1].strip() if 'vs' in matchup_text else "Unknown"
                
                props.append({
                    'player': player,
                    'line': line,
                    'prop_type': prop_type,
                    'over_odds': over_odds,
                    'under_odds': under_odds,
                    'opponent': opponent,
                    'is_home': is_home,
                    'bookmaker': 'BetOnline',
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                print(f"Erreur parsing prop: {e}")
                continue
        
        return props
    
    def _get_fallback_data(self):
        """
        Données de secours si BetOnline n'est pas accessible
        Simule des lignes réalistes
        """
        return [
            {
                'player': 'LeBron James',
                'line': 24.5,
                'prop_type': 'Points',
                'over_odds': '-110',
                'under_odds': '-110',
                'opponent': 'Warriors',
                'is_home': True,
                'bookmaker': 'BetOnline (Simulated)',
                'timestamp': datetime.now().isoformat()
            },
            {
                'player': 'Stephen Curry',
                'line': 27.5,
                'prop_type': 'Points',
                'over_odds': '-115',
                'under_odds': '-105',
                'opponent': 'Lakers',
                'is_home': False,
                'bookmaker': 'BetOnline (Simulated)',
                'timestamp': datetime.now().isoformat()
            },
            {
                'player': 'Luka Doncic',
                'line': 29.5,
                'prop_type': 'Points',
                'over_odds': '-110',
                'under_odds': '-110',
                'opponent': 'Suns',
                'is_home': True,
                'bookmaker': 'BetOnline (Simulated)',
                'timestamp': datetime.now().isoformat()
            },
            {
                'player': 'Kevin Durant',
                'line': 26.5,
                'prop_type': 'Points',
                'over_odds': '-110',
                'under_odds': '-110',
                'opponent': 'Celtics',
                'is_home': False,
                'bookmaker': 'BetOnline (Simulated)',
                'timestamp': datetime.now().isoformat()
            },
            {
                'player': 'Giannis Antetokounmpo',
                'line': 30.5,
                'prop_type': 'Points',
                'over_odds': '-105',
                'under_odds': '-115',
                'opponent': 'Heat',
                'is_home': True,
                'bookmaker': 'BetOnline (Simulated)',
                'timestamp': datetime.now().isoformat()
            },
            {
                'player': 'Jayson Tatum',
                'line': 26.5,
                'prop_type': 'Points',
                'over_odds': '-110',
                'under_odds': '-110',
                'opponent': 'Nets',
                'is_home': True,
                'bookmaker': 'BetOnline (Simulated)',
                'timestamp': datetime.now().isoformat()
            },
            {
                'player': 'Nikola Jokic',
                'line': 25.5,
                'prop_type': 'Points',
                'over_odds': '-110',
                'under_odds': '-110',
                'opponent': 'Clippers',
                'is_home': False,
                'bookmaker': 'BetOnline (Simulated)',
                'timestamp': datetime.now().isoformat()
            },
            {
                'player': 'Joel Embiid',
                'line': 31.5,
                'prop_type': 'Points',
                'over_odds': '-110',
                'under_odds': '-110',
                'opponent': 'Bucks',
                'is_home': True,
                'bookmaker': 'BetOnline (Simulated)',
                'timestamp': datetime.now().isoformat()
            }
        ]
    
    def get_available_players(self):
        """
        Retourne la liste des joueurs disponibles sur BetOnline
        """
        props = self.get_nba_player_props()
        players = list(set([p['player'] for p in props]))
        return sorted(players)
    
    def get_player_line(self, player_name):
        """
        Obtient la ligne spécifique pour un joueur
        """
        props = self.get_nba_player_props()
        
        for prop in props:
            if prop['player'].lower() == player_name.lower():
                return prop
        
        return None

# Test du scraper
if __name__ == '__main__':
    scraper = BetOnlineScraper()
    
    print("🎲 Test du scraper BetOnline...")
    print("=" * 50)
    
    props = scraper.get_nba_player_props()
    
    print(f"\n✅ {len(props)} props disponibles:\n")
    
    for prop in props:
        print(f"👤 {prop['player']}")
        print(f"   Ligne: {prop['line']} points")
        print(f"   Matchup: {'vs' if prop['is_home'] else '@'} {prop['opponent']}")
        print(f"   Odds: Over {prop['over_odds']} / Under {prop['under_odds']}")
        print(f"   Source: {prop['bookmaker']}")
        print()
