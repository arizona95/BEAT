import numpy as np
import pickle

class System :
    def __init__(self, model):
        # shape of parameter
        self.n   = model["n"]
        self.c   = model["c"]
        self.e   = model["e"]

        # system parameter
        self.x_0 = model["x_0"]
        self.p_0 = model["p_0"]
        self.M   = model["M"]
        self.M_  = model["M_"]
        self.S   = model["S"]
        self.D   = model["D"]
        self.m_c = model["m_c"]
        self.q_c = model["q_c"]
        self.a   = model["a"]
        self.k   = model["k"]
        self.v   = model["v"]
        self.c   = model["c"]
        self.x_h = model["x_h"]
        self.h   = model["h"]

        self.assert_parameter()
        self.initial_calculate()


    @classmethod
    def import_model(cls, filename):
        # init from a file
        with open(filename, 'rb') as fr:
            model = pickle.load(fr)
            return cls(model)

    def assert_parameter(self,):
        # assert system parameter
        assert self.x_0.shape == (self.n, 1)
        assert (self.x_0 > 0).all()

    def initial_calculate(self,):
        # calculate initial parameter
        self.x, self.p = self.x_0, self.p_0
        self.q = np.dot(np.diag(np.dot(self.M, self.q_c).T[0]), self.S)
        self.V = np.dot(self.q, np.dot(1 / (self.D + 1) - 1, self.q.T))
        self.m = np.dot(self.M, self.m_c)

    def energy(self,):
        # energy of system
        energy =\
            np.dot(self.x.T, np.log(self.x)) +\
            np.dot(self.a.T, self.x) +\
            0.5 * np.dot(self.x.T, np.dot(self.V, self.x)) +\
            0.5 * (np.dot((1 / self.m.T), (self.p * self.p * self.x)))

        return energy[0]

    def flow(self,) :
        # gradient
        grad_x =\
            (np.log(self.x)+1).T +\
            self.a.T +\
            np.dot(self.x.T, self.V) +\
            (0.5*(1/self.m)*self.p*self.p).T
        grad_p = ((1/self.m)*self.p*self.x).T

        # flow
        chemical_flow_x =\
            -np.dot(self.M_,(self.k.T*\
                (np.exp(np.dot(grad_x, self.M_))-1)*\
                (np.exp(-0.5*np.dot(grad_x, self.M_)))*\
                (np.exp( 0.5*np.dot(np.log(self.x.T), np.abs(self.M_))))).T)
        hamiltonian_flow_x =np.dot(self.M_, self.v*(np.dot(grad_p, self.M_).T) )
        hamiltonian_flow_p = -np.dot(self.M_, self.v*(np.dot(grad_x, self.M_).T) )
        colision_flow_p = -self.c*(grad_p).T
        external_homeostasis_flow_x = self.h*(self.x_h-self.x)

        return chemical_flow_x + hamiltonian_flow_x + external_homeostasis_flow_x,  hamiltonian_flow_p + colision_flow_p

    def ode(self, xp, t):
        xp_2d = xp.reshape(2, -1)
        self.x = np.array([xp_2d[0]]).T
        self.p = np.array([xp_2d[1]]).T
        flow_x, flow_p = self.flow()
        dxpdt = np.array([flow_x, flow_p]).reshape(-1)
        return dxpdt

    def save_model(self):
        # save model
        model=dict()
        model["n"] = self.n
        model["c"] = self.c
        model["e"] = self.e
        model["x_0"] = self.x_0
        model["M"] = self.M
        model["M_"] = self.M_
        model["S"] = self.S
        model["D"] = self.D
        model["m_c"] = self.m_c
        model["q_c"] = self.q_c
        model["a"] = self.a
        model["k"] = self.k
        model["v"] = self.v
        model["c"] = self.c
        model["x_h"] = self.x_h
        model["h"] = self.h

        with open('model', 'wb') as fw:
            pickle.dump(model, fw)

