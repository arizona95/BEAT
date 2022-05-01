import numpy as np


class System:
    def __init__(self, model, mode="phy"):
        # mode
        self.mode = mode

        # shape of parameter
        self.n = model["n"]
        self.c = model["c"]
        self.e = model["e"]
        self.s = model["s"]

        # system parameter
        self.x_0 = model["x_0"]
        self.p_0 = model["p_0"]
        self.M = model["M"]
        self.M_ = model["M_"]
        self.S = model["S"]
        self.D = model["D"]
        self.m_c = model["m_c"]
        self.q_c = model["q_c"]
        self.a = model["a"]
        self.k = model["k"]
        self.v = model["v"]
        self.c_ = model["c_"]
        self.x_h = model["x_h"]
        self.h = model["h"]

        self.assert_parameter()
        self.initial_calculate()

        if self.mode == "bio":
            self.er = model["er"]
            self.ed = model["ed"]
            self.eh = model["eh"]
            self.setting_bio_model()

    def assert_parameter(self, ):
        # assert system parameter
        assert self.x_0.shape == (self.n, 1)
        assert (self.x_0 > 0).all()
        assert self.p_0.shape == (self.n, 1)
        assert self.M.shape == (self.n, self.c)
        assert (self.M >= 0).all()
        assert self.M_.shape == (self.n, self.e)
        assert (np.dot(self.M.T, self.M_) == 0).all()
        assert self.S.shape == (self.n, self.s)
        assert (self.S * (1 - self.S) == 0).all()
        assert (self.S.sum(axis=1) == 1).all()
        assert self.D.shape == (self.s, self.s)
        assert (self.D >= 0).all()
        assert (np.diag(self.D) == 0).all()
        assert self.m_c.shape == (self.c, 1)
        assert (self.m_c >= 0).all()
        assert self.q_c.shape == (self.c, 1)
        assert self.a.shape == (self.n, 1)
        assert self.k.shape == (self.e, 1)
        assert (self.k >= 0).all()
        assert self.v.shape == (self.e, 1)
        assert (self.v >= 0).all()
        assert self.c_.shape == (self.n, 1)
        assert (self.c_ >= 0).all()
        assert self.x_h.shape == (self.n, 1)
        assert (self.x_h >= 0).all()
        assert self.h.shape == (self.n, 1)
        assert (self.h >= 0).all()

    def setting_bio_model(self):

        def relu(x):
            return np.maximum(0, x)

        self.k_r = self.k[:self.er]
        self.k_d = self.k[self.er, :self.er + self.ed]
        self.M_r = self.M_[:, :self.er]
        self.M_d = self.M_[:, self.er:self.er + self.ed]
        self.M_h = self.M_[:, self.er + self.ed:self.er + self.ed + self.eh]
        assert self.er + self.ed + self.eh == self.e

        self.reactant1_sparse = np.zeros(self.M_r.shape)
        self.reactant1_sparse[np.arange(self.M_r.shape[1]), np.argmax(relu(self.M_r), axis=0)] = 1
        self.reactant2_sparse = relu(self.M_r) - self.reactant1_sparse
        self.reactant1 = self.reactant1_sparse.argmax(axis=0)
        self.reactant2 = self.reactant2_sparse.argmax(axis=0)
        self.product = relu(-self.M_r).argmax(axis=0)
        self.diff1 = relu(self.M_d).argmax(axis=0)
        self.diff2 = relu(-self.M_d).argmax(axis=0)


    def initial_calculate(self, ):
        # calculate initial parameter
        self.x, self.p = self.x_0, self.p_0
        self.q = np.dot(np.diag(np.dot(self.M, self.q_c).T[0]), self.S)
        self.V = np.dot(self.q, np.dot(1 / (self.D + 1) - 1, self.q.T))
        self.m = np.dot(self.M, self.m_c)

    def energy(self, ):
        # energy of system
        energy = \
            np.dot(self.x.T, np.log(self.x)) + \
            np.dot(self.a.T, self.x) + \
            0.5 * np.dot(self.x.T, np.dot(self.V, self.x)) + \
            0.5 * (np.dot((1 / self.m.T), (self.p * self.p * self.x)))

        return energy[0]

    def flow(self, mode="phy"):
        # gradient
        # mode : phy, bio
        if self.mode == "phy":
            grad_x = \
                (np.log(self.x) + 1).T + \
                self.a.T + \
                np.dot(self.x.T, self.V) + \
                (0.5 * (1 / self.m) * self.p * self.p).T
            grad_p = ((1 / self.m) * self.p * self.x).T

            # flow
            chemical_flow_x = \
                -np.dot(self.M_, (self.k.T * \
                                  (np.exp(np.dot(grad_x, self.M_)) - 1) * \
                                  (np.exp(-0.5 * np.dot(grad_x, self.M_))) * \
                                  (np.exp(0.5 * np.dot((np.log(self.x) + 1 + self.a).T, np.abs(self.M_))))).T)
            hamiltonian_flow_x = np.dot(self.M_, self.v * \
                                        (np.dot(grad_p, self.M_).T) * \
                                        (np.exp(np.dot(np.abs(self.M_.T), np.log(self.x)))))
            hamiltonian_flow_p = -np.dot(self.M_, self.v * \
                                         (np.dot(grad_x, self.M_).T) * \
                                         (np.exp(np.dot(np.abs(self.M_.T), np.log(self.x)))))
            colision_flow_p = -self.c_ * (grad_p).T
            external_homeostasis_flow_x = self.h * (self.x_h - self.x)

            return chemical_flow_x + hamiltonian_flow_x + external_homeostasis_flow_x, hamiltonian_flow_p + colision_flow_p

        elif self.mode == "bio":

            grad_x = \
                (np.log(self.x) + 1).T + \
                self.a.T + \
                np.dot(self.x.T, self.V) + \
                (0.5 * (1 / self.m) * self.p * self.p).T
            grad_p = ((1 / self.m) * self.p * self.x).T

            self.x_a_cal = np.exp(grad_x).T

            # flow
            chemical_flow_x = \
                -np.dot(self.M_r, (self.k_r * \
                                   (-np.apply_along_axis(lambda x: x[self.product], 0, self.x_a_cal) + \
                                    np.apply_along_axis(lambda x: x[self.reactant1], 0, self.x_a_cal) * \
                                    np.apply_along_axis(lambda x: x[self.reactant2], 0, self.x_a_cal))))
            diffusion_flow_x = -np.dot(self.M_d, (self.k_d * \
                                  (-np.apply_along_axis(lambda x: x[self.diff1], 0, self.x_a_cal) + \
                                   np.apply_along_axis(lambda x: x[self.diff2], 0, self.x_a_cal))))

            hamiltonian_flow_x = np.dot(self.M_, self.v * \
                                        (np.dot(grad_p, self.M_).T) * \
                                        (np.exp(np.dot(np.abs(self.M_.T), np.log(self.x)))))
            hamiltonian_flow_p = -np.dot(self.M_, self.v * \
                                         (np.dot(grad_x, self.M_).T) * \
                                         (np.exp(np.dot(np.abs(self.M_.T), np.log(self.x)))))
            colision_flow_p = -self.c_ * (grad_p).T
            external_homeostasis_flow_x = self.h * (self.x_h - self.x)

            return chemical_flow_x + hamiltonian_flow_x + external_homeostasis_flow_x, hamiltonian_flow_p + colision_flow_p

    def ode(self, t, xp):
        xp_2d = xp.reshape(2, -1)
        self.x = np.array([xp_2d[0]]).T
        self.p = np.array([xp_2d[1]]).T
        flow_x, flow_p = self.flow()
        dxpdt = np.array([flow_x, flow_p]).reshape(-1)
        return dxpdt


