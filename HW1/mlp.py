import torch

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        # TODO: Implement the forward function
        W1, b1, W2, b2 = self.parameters['W1'], self.parameters['b1'], self.parameters['W2'], self.parameters['b2']
        z1 = torch.matmul(x, W1.T) + b1

        if self.f_function == "relu":
            a1 = torch.relu(z1)
        elif self.f_function == "sigmoid":
            a1 = torch.sigmoid(z1)
        else:
            a1 = z1

        z2 = torch.matmul(a1, W2.T) + b2

        if self.g_function == "relu":
            y_hat = torch.relu(z2)
        elif self.g_function == "sigmoid":
            y_hat = torch.sigmoid(z2)
        else:
            y_hat = z2
        
        self.cache['y_hat'] = y_hat
        self.cache['z1'] = z1
        self.cache['a1'] = a1
        self.cache['z2'] = z2
        self.cache['x'] = x
        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        W2 = self.parameters['W2']
        a1, z1, z2 = self.cache['a1'], self.cache['z1'], self.cache['z2']
        
        if self.g_function == "relu":
            dJdz2 = dJdy_hat * (z2 > 0).float()  
        elif self.g_function == "sigmoid":
            sig_z2 = torch.sigmoid(z2)
            dJdz2 = dJdy_hat * sig_z2 * (1 - sig_z2)
        else:  
            dJdz2 = dJdy_hat  
        

        self.grads['dJdW2'] = torch.matmul(dJdz2.T, a1) / dJdy_hat.shape[1]
        self.grads['dJdb2'] = dJdz2.sum(dim=0) / dJdy_hat.shape[1]
        
        dJda1 = torch.matmul(dJdz2, W2)

        if self.f_function == "relu":
            dJdz1 = dJda1 * (z1 > 0).float()
        elif self.f_function == "sigmoid":
            sig_z1 = torch.sigmoid(z1)
            dJdz1 = dJda1 * sig_z1 * (1 - sig_z1)
        else:
            dJdz1 = dJda1

        self.grads['dJdW1'] = torch.matmul(dJdz1.T, self.cache['x']) / dJdy_hat.shape[1]
        self.grads['dJdb1'] = dJdz1.sum(dim=0) / dJdy_hat.shape[1]

    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    loss = torch.sum((y - y_hat) ** 2)
    dJdy_hat = (2 * (y_hat - y)) / y.shape[0]
    return loss, dJdy_hat
    

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    loss = -torch.mean(y * torch.log(y_hat) + (1 - y) * torch.log(1 + y_hat)) 
    dJdy_hat = (y_hat - y) / (y_hat * (1 - y_hat)) 
    return loss, dJdy_hat











