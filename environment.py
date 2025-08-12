import gymnasium as gym
import numpy as np
import random
from collections import deque
from typing import List, Tuple

class SportkaEnvironment(gym.Env):
    """
    Prostředí pro Sportku s historickými daty a superjackpotem
    """
    
    def __init__(self, max_episodes=1000):
        super(SportkaEnvironment, self).__init__()
        
        # Sportka parametry
        self.min_number = 1
        self.max_number = 49
        self.numbers_per_draw = 6
        self.ticket_cost = 30  # Kč za sloupeček
        self.chance_cost = 30  # Kč za Šanci
        
        # Observation space: 5 posledních losování (5x6 čísel) + aktuální balance + draw counter
        # Každé losování je reprezentováno jako 49-dimenzionální binary vektor
        self.observation_space = gym.spaces.Box(
            low=0, high=1, 
            shape=(5 * 49 + 2,),  # 5 losování * 49 čísel + balance + draw_counter
            dtype=np.float32
        )
        
        # Action space: 6 čísel (1-49) + Šance (0/1)
        # Reprezentujeme jako 49-dimenzionální binary vektor pro čísla + 1 bit pro Šanci
        self.action_space = gym.spaces.Box(
            low=0, high=1,
            shape=(50,),  # 49 čísel + Šance
            dtype=np.float32
        )
        
        # Historie losování (uchovávání posledních 5)
        self.draw_history = deque(maxlen=5)
        self.max_episodes = max_episodes
        self.reset()
    
    def _generate_realistic_draw(self) -> Tuple[List[int], int]:
        """
        Generuje realistické losování s ohledem na historii
        """
        # Základní losování
        drawn_numbers = sorted(random.sample(range(1, 50), 6))
        chance_number = random.randint(0, 9)
        
        # Pokud máme historii, snižujeme pravděpodobnost opakování
        if len(self.draw_history) > 0:
            recent_numbers = set()
            for hist_draw in list(self.draw_history)[-2:]:  # Poslední 2 losování
                recent_numbers.update(hist_draw[0])
            
            # Pokud se číslo vyskytlo v posledních 2 losování, 80% šance na nové číslo
            for i, num in enumerate(drawn_numbers):
                if num in recent_numbers and random.random() < 0.8:
                    # Najdi náhradní číslo, které nebylo nedávno
                    available = [n for n in range(1, 50) if n not in recent_numbers and n not in drawn_numbers]
                    if available:
                        drawn_numbers[i] = random.choice(available)
            
            drawn_numbers = sorted(drawn_numbers)
        
        return drawn_numbers, chance_number
    
    def _encode_draw(self, numbers: List[int]) -> np.ndarray:
        """Kóduje losování jako binary vektor"""
        encoded = np.zeros(49, dtype=np.float32)
        for num in numbers:
            encoded[num - 1] = 1.0
        return encoded
    
    def _decode_action(self, action: np.ndarray) -> Tuple[List[int], bool]:
        """Dekóduje akci na čísla a Šanci"""
        # Získej 6 největších hodnot pro čísla
        numbers_probs = action[:49]
        selected_indices = np.argsort(numbers_probs)[-6:]  # 6 nejvyšších
        selected_numbers = sorted([idx + 1 for idx in selected_indices])
        
        # Šance (poslední element)
        play_chance = action[49] > 0.5
        
        return selected_numbers, play_chance
    
    def _calculate_winnings(self, player_numbers: List[int], drawn_numbers: List[int], 
                          player_chance: bool, drawn_chance: int, 
                          last_digit_match: bool) -> float:
        """Vypočítá výhru podle pravidel Sportky"""
        matches = len(set(player_numbers) & set(drawn_numbers))
        
        # Základní výherní tabulka (zjednodušená)
        winnings = 0.0
        if matches == 3:
            winnings = 50.0
        elif matches == 4:
            winnings = 300.0
        elif matches == 5:
            winnings = 10000.0
        elif matches == 6:
            winnings = 100000.0  # Jackpot
            
            # Superjackpot bonus
            if player_chance and last_digit_match:
                winnings += 1000000.0  # Superjackpot bonus
        
        # Šance výhry (zjednodušeno)
        if player_chance:
            if last_digit_match:
                winnings += 1000.0  # Bonus za Šanci
        
        return winnings
    
    def _get_state(self) -> np.ndarray:
        """Získá aktuální stav prostředí"""
        state = []
        
        # Encode posledních 5 losování
        for i in range(5):
            if i < len(self.draw_history):
                idx = len(self.draw_history) - 1 - i  # Od nejnovějšího k nejstaršímu
                encoded_draw = self._encode_draw(self.draw_history[idx][0])
            else:
                encoded_draw = np.zeros(49, dtype=np.float32)
            state.extend(encoded_draw)
        
        # Přidej balance a draw counter (normalizované)
        state.append(self.balance / 10000.0)  # Normalizace balance
        state.append(self.current_draw / self.max_episodes)  # Normalizace draw counter
        
        return np.array(state, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        """Reset prostředí"""
        super().reset(seed=seed)
        
        self.balance = 30000.0  # Počáteční balance pro 1000 tiketů
        self.current_draw = 0
        self.total_winnings = 0.0
        self.total_spent = 0.0
        self.tickets_bought = 0
        self.draw_history.clear()
        
        # Generuj počáteční historii
        for _ in range(5):
            drawn_numbers, chance_number = self._generate_realistic_draw()
            self.draw_history.append((drawn_numbers, chance_number))
        
        return self._get_state(), {}
    
    def step(self, action):
        """Provede krok v prostředí"""
        player_numbers, player_chance = self._decode_action(action)
        
        # Vypočítej náklady
        cost = self.ticket_cost
        if player_chance:
            cost += self.chance_cost
        
        # Kontrola, zda má hráč dostatek peněz
        if self.balance < cost:
            # Nemá dostatek peněz - konec hry
            reward = -100.0  # Penalizace za nedostatek prostředků
            return self._get_state(), reward, True, True, {
                "reason": "insufficient_funds",
                "balance": self.balance,
                "total_winnings": self.total_winnings,
                "total_spent": self.total_spent,
                "tickets_bought": self.tickets_bought
            }
        
        # Odečti náklady
        self.balance -= cost
        self.total_spent += cost
        self.tickets_bought += 1
        
        # Generuj nové losování
        drawn_numbers, drawn_chance = self._generate_realistic_draw()
        self.draw_history.append((drawn_numbers, drawn_chance))
        
        # Kontrola posledního čísla pro Superjackpot
        # Simulujeme, že poslední číslo tiketu je odvozeno od Šance
        last_digit_match = (drawn_chance == (sum(player_numbers) % 10))
        
        # Vypočítej výhru
        winnings = self._calculate_winnings(
            player_numbers, drawn_numbers, player_chance, 
            drawn_chance, last_digit_match
        )
        
        self.balance += winnings
        self.total_winnings += winnings
        
        # Vypočítej reward
        net_gain = winnings - cost
        reward = net_gain / 100.0  # Škálování
        
        # Bonus za historicky informované rozhodování
        matches = len(set(player_numbers) & set(drawn_numbers))
        if matches >= 3:
            # Bonus za vyhranou
            reward += matches * 10.0
        
        # Penalizace za opakování nedávných čísel
        recent_numbers = set()
        if len(self.draw_history) >= 2:
            for hist_draw in list(self.draw_history)[-2:-1]:  # Předchozí losování
                recent_numbers.update(hist_draw[0])
        
        repeated = len(set(player_numbers) & recent_numbers)
        if repeated > 2:  # Více než 2 opakující se čísla
            reward -= repeated * 2.0
        
        self.current_draw += 1
        
        # Kontrola ukončení - končíme po 1000 tiketech nebo při nedostatku peněz
        terminated = (self.tickets_bought >= 1000) or (self.balance < self.ticket_cost)
        
        info = {
            "player_numbers": player_numbers,
            "drawn_numbers": drawn_numbers,
            "matches": matches,
            "winnings": winnings,
            "balance": self.balance,
            "play_chance": player_chance,
            "drawn_chance": drawn_chance,
            "last_digit_match": last_digit_match,
            "net_gain": net_gain,
            "total_winnings": self.total_winnings,
            "total_spent": self.total_spent,
            "tickets_bought": self.tickets_bought
        }
        
        return self._get_state(), reward, terminated, False, info