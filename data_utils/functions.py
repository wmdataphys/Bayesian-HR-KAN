import numpy as np

TYPE_ = "Student"
DOF_ = 3

def f1(x):
    return np.sin(np.pi * x)

def f1_noise(x,sigma=0.1,dof=DOF_,type_=TYPE_,return_clean=False):
    if type_ == "Gauss":
        if return_clean: 
            return np.sin(np.pi * x) + np.random.normal(loc=0,scale=sigma,size=x.shape), np.sin(np.pi * x)
        else:
            return np.sin(np.pi * x) + np.random.normal(loc=0,scale=sigma,size=x.shape)
    elif type_ == "Student":
        if return_clean:
            return np.sin(np.pi * x) + np.random.standard_t(df=dof,size=x.shape),np.sin(np.pi*x)
        else:
            return np.sin(np.pi * x) + np.random.standard_t(df=dof,size=x.shape)
    else:
        raise ValueError('Noise not implemented.')


def f2(x):
    return np.sin(5*np.pi*x) + x

def f2_noise(x,sigma=0.1,dof=DOF_,type_=TYPE_,return_clean=False):
    if type_ == "Gauss":
        if return_clean:
            return np.sin(5*np.pi*x) + x + np.random.normal(loc=0,scale=sigma,size=x.shape), np.sin(5*np.pi*x) + x
        else:
            return np.sin(5*np.pi*x) + x + np.random.normal(loc=0,scale=sigma,size=x.shape)
    elif type_ == "Student":
        if return_clean:
            return np.sin(5*np.pi*x) + x +  np.random.standard_t(df=dof,size=x.shape), np.sin(5*np.pi*x) + x
        else:
            return np.sin(5*np.pi*x) + x +  np.random.standard_t(df=dof,size=x.shape)
    else:
        raise ValueError("Noise not implemented.")

def f3(x):
    return np.exp(x)

def f3_noise(x,sigma=0.1,dof=DOF_,type_=TYPE_,return_clean=False):
    if type_ == "Gauss":
        if return_clean:
            return np.exp(x) + np.random.normal(loc=0,scale=sigma,size=x.shape),np.exp(x)
        else:
            return np.exp(x) + np.random.normal(loc=0,scale=sigma,size=x.shape)
    elif type_ == "Student":
        if return_clean:
            return np.exp(x) + np.random.standard_t(df=dof,size=x.shape),np.exp(x)
        else:
            return np.exp(x) + np.random.standard_t(df=dof,size=x.shape)
    else:
        raise ValueError("Noise not implemented.")

def f4(x):
    y = np.sin(np.pi * x[:, [0]] + np.pi * x[:, [1]]) 
    return y


def f4_noise(x,sigma=0.1):
    y = np.sin(np.pi * x[:, [0]] + np.pi * x[:, [1]]) + np.random.normal(loc=0,scale=sigma,size=x.shape)
    return y

def f5(x):
    y = np.exp(np.sin(np.pi * x[:, [0]]) + x[:, [1]] * x[:, [1]])
    return y


def f6(x):
    y = np.exp(
        np.sin(np.pi * x[:, [0]] * x[:, [0]] + np.pi * x[:, [1]] * x[:, [1]]) +
        np.sin(np.pi * x[:, [2]] * x[:, [2]] + np.pi * x[:, [3]] * x[:, [3]])
    )
    return y

def f7(x):
    return np.sin(5 * np.pi * x) + x
