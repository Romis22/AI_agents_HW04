import numpy as np
import matplotlib.pyplot as plt
from environment import SportkaEnvironment
from models import DQNAgent

def train_agent(episodes=2000, save_path="sportka_model.pth"):
    """Trénuje DQN agenta na Sportce"""
    env = SportkaEnvironment(max_episodes=1000)  # Max 1000 tiketů
    agent = DQNAgent(env.observation_space.shape[0], env.action_space.shape[0])
    
    scores = []
    balances = []
    winnings = []
    tickets = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        step_count = 0
        
        while True:
            action = agent.act(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            agent.remember(state, action, reward, next_state, terminated)
            state = next_state
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
        
        scores.append(total_reward)
        balances.append(info['balance'])
        winnings.append(info['total_winnings'])
        tickets.append(info['tickets_bought'])
        
        # Train agent
        if len(agent.memory) > agent.batch_size:
            agent.replay()
        
        # Update target network every 100 episodes
        if episode % 100 == 0:
            agent.update_target_network()
        
        # Výpis progressu
        if episode % 100 == 0:
            avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
            avg_balance = np.mean(balances[-100:]) if len(balances) >= 100 else np.mean(balances)
            avg_winnings = np.mean(winnings[-100:]) if len(winnings) >= 100 else np.mean(winnings)
            avg_tickets = np.mean(tickets[-100:]) if len(tickets) >= 100 else np.mean(tickets)
            
            print(f"Episode {episode}, Avg Score: {avg_score:.2f}, "
                  f"Avg Balance: {avg_balance:.2f} Kč, Avg Winnings: {avg_winnings:.2f} Kč, "
                  f"Avg Tickets: {avg_tickets:.0f}, Epsilon: {agent.epsilon:.3f}")
    
    # Uložení modelu
    agent.save(save_path)
    
    # Vizualizace výsledků
    plot_training_results(scores, balances, winnings, tickets)
    
    return agent, scores, balances, winnings

def plot_training_results(scores, balances, winnings, tickets):
    """Vykreslí výsledky trénování"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Skóre
    ax1.plot(scores, alpha=0.7)
    ax1.set_title('Training Scores')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Score')
    ax1.grid(True, alpha=0.3)
    
    # Balance
    ax2.plot(balances, alpha=0.7)
    ax2.axhline(y=30000, color='r', linestyle='--', alpha=0.5, label='Počáteční balance')
    ax2.set_title('Final Balance per Episode')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Balance (Kč)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ROI
    roi = [(w - 30000) / 30000 * 100 for w in winnings]
    ax3.plot(roi, alpha=0.7)
    ax3.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
    ax3.set_title('Return on Investment (ROI)')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('ROI (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Moving average ROI
    window_size = 50
    if len(roi) >= window_size:
        moving_avg = []
        for i in range(window_size-1, len(roi)):
            moving_avg.append(np.mean(roi[i-window_size+1:i+1]))
        ax4.plot(range(window_size-1, len(roi)), moving_avg, color='green')
        ax4.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Break-even')
        ax4.set_title(f'Moving Average ROI (window={window_size})')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('ROI (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Training Results - DQN Agent for Sportka', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    print("=== SPORTKA REINFORCEMENT LEARNING - TRÉNOVÁNÍ ===")
    print("Trénování DQN agenta pro optimální strategii Sportky...")
    print("Agent má k dispozici 30,000 Kč (1000 tiketů po 30 Kč)")
    print("Agent se učí na základě posledních 5 losování a používá historické informace.")
    print("=" * 60)
    print()
    
    # Spustit trénování
    agent, scores, balances, winnings = train_agent(episodes=2000, save_path="sportka_model.pth")
    
    print("\n" + "=" * 60)
    print("Trénování dokončeno!")
    print("Model uložen jako 'sportka_model.pth'")
    print("Grafy uloženy jako 'training_results.png'")
    print("\nPro testování spusťte: python main.py")