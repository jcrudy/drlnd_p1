
class FixedQTargetModel(Model):
    def __init__(self, network_maker, optimizer_maker, gamma):
        self.q_local = self.network_maker()
        self.q_target = self.network_maker()
        self.optimizer = optimizer_maker(self.q_local)
        self.gamma = gamma
        
    def learn(self, state, action, reward, next_state, weight):
        q_target_output = self.q_target.forward(next_state).detach().max(1)[0].unsqueeze(1)
        target = reward + self.gamma * q_target_output
        prediction = self.q_local.forward(state)
        
        loss = F.mse_loss(prediction, target)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()