from .loss import Loss


class L2(Loss):
    def forward(self, x1, x1_hat, x1_m_x0, x1_m_x0_hat):
        return ((x1_m_x0 - x1_m_x0_hat) ** 2).mean([-1, -2, -3])

    @property
    def name(self):
        return "l2"
