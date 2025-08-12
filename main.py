import numpy as np
import matplotlib.pyplot as plt
from environment import SportkaEnvironment
from models import DQNAgent, RandomAgent
import os
from scipy import stats

def test_agent(agent, agent_name="Agent", num_runs=10):
    """
    Testuje agenta na 1000 sázkách s 30,000 Kč
    
    Args:
        agent: Agent k testování
        agent_name: Název agenta pro výpis
        num_runs: Počet testovacích běhů
    
    Returns:
        Dictionary s výsledky
    """
    env = SportkaEnvironment(max_episodes=1000)
    
    results = {
        'name': agent_name,
        'final_balances': [],
        'total_winnings': [],
        'total_spent': [],
        'tickets_bought': [],
        'roi': [],
        'win_counts': {3: 0, 4: 0, 5: 0, 6: 0},
        'chance_played': 0,
        'chance_wins': 0
    }
    
    for run in range(num_runs):
        state, _ = env.reset()
        run_win_counts = {3: 0, 4: 0, 5: 0, 6: 0}
        
        while True:
            # Získej akci podle typu agenta
            if isinstance(agent, DQNAgent):
                action = agent.act(state, training=False)
            else:  # RandomAgent
                action = agent.act()
            
            next_state, _, terminated, truncated, info = env.step(action)
            
            # Počítej výhry
            matches = info['matches']
            if matches >= 3:
                run_win_counts[matches] += 1
            
            # Počítej Šanci
            if info['play_chance']:
                results['chance_played'] += 1
                if info['last_digit_match']:
                    results['chance_wins'] += 1
            
            state = next_state
            
            if terminated or truncated:
                results['final_balances'].append(info['balance'])
                results['total_winnings'].append(info['total_winnings'])
                results['total_spent'].append(info['total_spent'])
                results['tickets_bought'].append(info['tickets_bought'])
                roi = ((info['total_winnings'] - info['total_spent']) / info['total_spent']) * 100
                results['roi'].append(roi)
                
                # Přidej výhry do celkových
                for key in run_win_counts:
                    results['win_counts'][key] += run_win_counts[key]
                
                print(f"  Run {run+1}/{num_runs}: Balance: {info['balance']:.0f} Kč, "
                      f"Winnings: {info['total_winnings']:.0f} Kč, "
                      f"ROI: {roi:.2f}%, Tickets: {info['tickets_bought']}")
                break
    
    return results

def compare_agents(trained_agent, num_runs=10):
    """
    Porovnává natrénovaného agenta s náhodným agentem
    
    Args:
        trained_agent: Natrénovaný DQN agent
        num_runs: Počet testovacích běhů pro každého agenta
    """
    print("\n" + "=" * 80)
    print("TESTOVÁNÍ A POROVNÁNÍ AGENTŮ")
    print(f"Každý agent má 30,000 Kč a sází 1000 tiketů (nebo dokud má peníze)")
    print(f"Každý test se opakuje {num_runs}x pro statistickou významnost")
    print("=" * 80)
    
    # Test natrénovaného agenta
    print("\n1. TESTOVÁNÍ DQN AGENTA (natrénovaný)")
    print("-" * 40)
    dqn_results = test_agent(trained_agent, "DQN Agent", num_runs)
    
    # Test náhodného agenta
    print("\n2. TESTOVÁNÍ NÁHODNÉHO AGENTA")
    print("-" * 40)
    random_agent = RandomAgent()
    random_results = test_agent(random_agent, "Random Agent", num_runs)
    
    # Výpočet statistik
    print("\n" + "=" * 80)
    print("SOUHRNNÉ VÝSLEDKY")
    print("=" * 80)
    
    for results in [dqn_results, random_results]:
        print(f"\n{results['name']}:")
        print("-" * 40)
        print(f"  Průměrný konečný zůstatek: {np.mean(results['final_balances']):.2f} Kč")
        print(f"  Průměrné výhry: {np.mean(results['total_winnings']):.2f} Kč")
        print(f"  Průměrné výdaje: {np.mean(results['total_spent']):.2f} Kč")
        print(f"  Průměrné ROI: {np.mean(results['roi']):.2f}%")
        print(f"  Std. odchylka ROI: {np.std(results['roi']):.2f}%")
        print(f"  Nejlepší ROI: {np.max(results['roi']):.2f}%")
        print(f"  Nejhorší ROI: {np.min(results['roi']):.2f}%")
        print(f"  Průměrný počet tiketů: {np.mean(results['tickets_bought']):.0f}")
        
        print(f"\n  Počty výher (celkem za {num_runs} běhů):")
        total_tickets = sum(results['tickets_bought'])
        for matches, count in results['win_counts'].items():
            if count > 0:
                print(f"    {matches} shod: {count}x (pravděpodobnost: {count/total_tickets*100:.4f}%)")
        
        if results['chance_played'] > 0:
            print(f"\n  Šance statistiky:")
            print(f"    Zahrána: {results['chance_played']}x")
            print(f"    Výhry: {results['chance_wins']}x")
            print(f"    Úspěšnost: {results['chance_wins']/results['chance_played']*100:.2f}%")
    
    # Vizualizace porovnání
    visualize_comparison(dqn_results, random_results)
    
    # Statistické porovnání
    print("\n" + "=" * 80)
    print("STATISTICKÉ POROVNÁNÍ")
    print("=" * 80)
    
    dqn_roi_mean = np.mean(dqn_results['roi'])
    random_roi_mean = np.mean(random_results['roi'])
    
    improvement = ((dqn_roi_mean - random_roi_mean) / abs(random_roi_mean)) * 100 if random_roi_mean != 0 else 0
    
    print(f"\nROI Porovnání:")
    print(f"  DQN Agent:    {dqn_roi_mean:.2f}%")
    print(f"  Random Agent: {random_roi_mean:.2f}%")
    print(f"  Relativní zlepšení: {improvement:.2f}%")
    
    # T-test pro statistickou významnost
    t_stat, p_value = stats.ttest_ind(dqn_results['roi'], random_results['roi'])
    print(f"\nStatistická významnost (t-test):")
    print(f"  t-statistika: {t_stat:.4f}")
    print(f"  p-hodnota: {p_value:.4f}")
    if p_value < 0.05:
        print(f"  Závěr: Rozdíl je statisticky významný (p < 0.05)")
    else:
        print(f"  Závěr: Rozdíl není statisticky významný (p >= 0.05)")
    
    # Určení vítěze
    print("\n" + "=" * 80)
    print("ZÁVĚR")
    print("=" * 80)
    
    if dqn_roi_mean > random_roi_mean:
        print(f"✓ DQN Agent je lepší než náhodné sázení o {abs(improvement):.2f}%")
    elif random_roi_mean > dqn_roi_mean:
        print(f"✗ Náhodné sázení je lepší než DQN Agent o {abs(improvement):.2f}%")
    else:
        print("= Oba přístupy mají stejný výsledek")
    
    print(f"\nPoznámka: Oba přístupy jsou v dlouhodobém průměru ztrátové,")
    print(f"což odpovídá realitě loterií. DQN agent se snaží minimalizovat ztráty.")

def visualize_comparison(dqn_results, random_results):
    """Vytvoří vizualizaci porovnání agentů"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # ROI distribuce
    ax1.hist(dqn_results['roi'], bins=20, alpha=0.7, label='DQN Agent', color='blue')
    ax1.hist(random_results['roi'], bins=20, alpha=0.7, label='Random Agent', color='red')
    ax1.axvline(np.mean(dqn_results['roi']), color='blue', linestyle='--', linewidth=2)
    ax1.axvline(np.mean(random_results['roi']), color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('ROI (%)')
    ax1.set_ylabel('Frekvence')
    ax1.set_title('Distribuce ROI')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Box plot ROI
    ax2.boxplot([dqn_results['roi'], random_results['roi']], labels=['DQN Agent', 'Random Agent'])
    ax2.set_ylabel('ROI (%)')
    ax2.set_title('Porovnání ROI')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Průměrné výhry
    labels = ['DQN Agent', 'Random Agent']
    winnings = [np.mean(dqn_results['total_winnings']), np.mean(random_results['total_winnings'])]
    spent = [np.mean(dqn_results['total_spent']), np.mean(random_results['total_spent'])]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax3.bar(x - width/2, winnings, width, label='Výhry', color='green', alpha=0.7)
    ax3.bar(x + width/2, spent, width, label='Výdaje', color='red', alpha=0.7)
    ax3.set_ylabel('Částka (Kč)')
    ax3.set_title('Průměrné výhry vs výdaje')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Počty výher
    matches_categories = [3, 4, 5, 6]
    dqn_wins = [dqn_results['win_counts'][m] for m in matches_categories]
    random_wins = [random_results['win_counts'][m] for m in matches_categories]
    
    x = np.arange(len(matches_categories))
    ax4.bar(x - width/2, dqn_wins, width, label='DQN Agent', color='blue', alpha=0.7)
    ax4.bar(x + width/2, random_wins, width, label='Random Agent', color='red', alpha=0.7)
    ax4.set_xlabel('Počet shod')
    ax4.set_ylabel('Počet výher')
    ax4.set_title('Celkový počet výher podle shod')
    ax4.set_xticks(x)
    ax4.set_xticklabels(matches_categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Porovnání DQN Agenta vs Náhodné sázení (1000 tiketů, 30,000 Kč)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('comparison_results.png', dpi=150)
    plt.show()

def main():
    """Hlavní funkce pro testování"""
    print("=" * 80)
    print("SPORTKA RL - TESTOVÁNÍ A POROVNÁNÍ")
    print("=" * 80)
    
    # Načtení natrénovaného modelu
    model_path = "sportka_model.pth"
    
    if not os.path.exists(model_path):
        print(f"\n⚠ Model '{model_path}' nebyl nalezen!")
        print("Nejprve spusťte trénování: python train.py")
        return
    
    print(f"\nNačítání modelu z '{model_path}'...")
    env = SportkaEnvironment()
    trained_agent = DQNAgent(env.observation_space.shape[0], env.action_space.shape[0])
    trained_agent.load(model_path)
    trained_agent.epsilon = 0  # Vypnout exploraci pro testování
    
    # Spuštění testů
    print("\nSpouštím testy...")
    print("Každý agent dostane 30,000 Kč (na 1000 tiketů po 30 Kč)")
    print("Test se opakuje 10x pro každého agenta pro statistickou přesnost\n")
    
    compare_agents(trained_agent, num_runs=10)
    
    print("\n" + "=" * 80)
    print("Testování dokončeno!")
    print("Výsledky uloženy jako 'comparison_results.png'")
    print("=" * 80)

if __name__ == "__main__":
    main()