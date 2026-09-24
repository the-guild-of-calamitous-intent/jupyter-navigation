import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ![](https://images.pexels.com/photos/269790/pexels-photo-269790.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=650&w=940)

    # Inertial Navigation System (INS)

    Kevin J. Walchko

    4 May 2019

    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Extended Kalman Filter

    ### Model

    $$
    \dot x(t) = f(x(t), u(t)) + w(t) \quad w(t) \sim \mathcal{N}(0,Q(t))\\
    z(t) = h(x(t)) + v(t) \quad v(t) \sim \mathcal{N}(0,R(t))
    $$

    ### Initialize

    $$
    \hat x(t_0) = E[x(t_x)] \\
    P(t_0) = Var[x(t_0)]
    $$

    ### Predict-Update

    $$
    \dot{\hat x}(t) = f(\hat x(t), u(t)) + K(t)(z(t)-h(\hat x(t))) \\
    \dot P(t)=F(t)P(t)+P(t)F(t)^T-K(t)H(t)P(t)+Q(t) \\
    K(t)=P(t)H(t)^T R(t)^{-1} \\
    F(t)=\frac{\partial f}{\partial x} \Bigr|_{\hat x(t), u(t)} \\
    H(t)=\frac{\partial h}{\partial x} \Bigr|_{\hat x(t)}
    $$

    ## References

    - [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter)
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo
    return


@app.cell
def _():
    # '%matplotlib inline' command supported automatically in marimo
    return


@app.cell
def _():
    import numpy as np # matrix manipulations
    from matplotlib import pyplot as plt

    return np, plt


@app.cell
def _():
    from math import sin, cos, atan2, pi, sqrt, asin
    from math import radians as deg2rad
    from math import degrees as rad2deg

    return atan2, cos, pi, sin, sqrt


@app.cell
def _():
    from squaternion import Quaternion

    return (Quaternion,)


@app.cell
def _():
    import pickle
    import pandas as pd

    return pd, pickle


@app.cell
def _():
    from pyrk import RK4
    import pyrk
    print(pyrk.__version__)
    return (RK4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Raw Data

    The data collected is just the IMU sitting still on a table. Let's find the biases so we can correct the data.
    """)
    return


@app.cell
def _():
    import os
    os.listdir()
    return


@app.cell
def _(pd, pickle):
    # fname = "inertial-navigation/Inertial_Navigation/accel-z-up.pickle"
    fname = "inertial-navigation/Inertial_Navigation/imu-rpi.2019-05-04-20:47:18.445193.pickle.bag"
    data = pickle.load(open(fname,"rb"))

    # data is an array of IMU messages
    def get(bag, key):
        data = {"x":[], "y":[],"z":[],"ts":[]}
        for (x,y,z),ts in bag[key]:
            data["x"].append(x)
            data["y"].append(y)
            data["z"].append(z)
            data["ts"].append(ts)
        return data

    adf = pd.DataFrame(get(data, "accel"))
    gdf = pd.DataFrame(get(data, "gyro"))
    mdf = pd.DataFrame(get(data, "mag"))
    return adf, data, gdf, mdf


@app.cell
def _(adf):
    adf.describe()
    return


@app.cell
def _(adf):
    print(f"Biases [x,y,z]: {adf['x'].mean()} {adf['y'].mean()} {adf['z'].mean() - 1.0}")
    return


@app.cell
def _(adf, gdf, mdf, pd):
    def remove_bias(ddf):
        df = pd.DataFrame()
        for key in ["x","y","z"]:
            df[key]=ddf[key] - ddf[key].mean()
        df["ts"] = ddf["ts"]
        return df

    madf = remove_bias(adf)
    mgdf = remove_bias(gdf)
    mmdf = remove_bias(mdf)
    return madf, mmdf


@app.cell
def _(madf):
    madf.plot(x="ts",subplots=True, grid=True);
    return


@app.cell
def _(mmdf):
    mmdf.plot(x="ts",subplots=True, grid=True);
    return


@app.cell
def _(mmdf):
    for x, _y, _z, t in mmdf.itertuples(index=False):
        print(x, _y, _z, t)
    return


@app.cell
def _(np, sqrt):
    def normalize(x, y, z):
        """Return a unit vector"""
        norm = sqrt(x * x + y * y + z * z)
        if norm == 1.0:
            return (x, y, z)  # already a unit vector
        if norm > 0.0:
            inorm = 1 / norm
            x = x * inorm
            y = y * inorm
            z = z * inorm
        else:
            raise Exception('division by zero: {} {} {}'.format(x, y, z))
        return (x, y, z)

    def skew4(wx, wy, wz):
        return np.array([(0, -wx, -wy, -wz), (wx, 0, wz, -wy), (wy, -wz, 0, wx), (wz, wy, -wx, 0)])

    def skew3(wx, wy, wz):  # titterton 11.35
        return np.array([(0, -wz, wy), (wz, 0, -wx), (-wy, wx, 0)])

    return (skew4,)


@app.cell
def _(cos, pi, sin, sqrt):
    def gps2ecef(lat, lon, H):
        e = 1.0  # phi = lat
        re = 6378137.0  # lambda = lon
        lat = lat * (pi / 180)  # H = height above mean sea-level (altitude)
        lon = lon * (pi / 180)
        rm = re * (1.0 - e ** 2) / pow(1.0 - e ** 2 * sin(lat) ** 2, 3.0 / 2.0)  # radius of Earth in meters
        rn = re / sqrt(1.0 - e ** 2 * sin(lat) ** 2)
        x = (rn + H) * cos(lat) * cos(lon)  # convert degrees to angles
        y = (rn + H) * cos(lat) * sin(lon)
        z = (rm + H) * sin(lat)
        return (x, y, z)

    return


@app.cell
def _(RK4, np):
    class Sim(object):
        def __init__(self, eom, a, g, tm):
            self.eom = eom
    
            # grab the inertial sensors
            self.accels = a
            self.gyros = g
            self.dts = tm

        def run_nav(self):
            """
            These are the navigation only equations, no EKF corrections
            """
            rk = RK4(self.eom)
    
            # initial states
            vel = (0,0,0)
            pos = (0,0,0)
            orient = (1,0,0,0)
            X = np.array(vel + pos + orient)
    
            save = []
    
            last = self.dts[0]-0.1
            for i, tm in enumerate(self.dts):
                u = self.accels[i] + self.gyros[i]
                X = rk.step(X,u,tm,tm-last)
                last = tm
                save.append(X)
        
            print(">> Simulation end: {:.2f} seconds".format(tm))
            print(">> Steps: {} at {:0.3f} sec".format(len(save), np.mean(np.diff(self.dts))))
        
            return save, self.dts

    return (Sim,)


@app.cell
def _(data, np, pi):
    accels = [(x[0][0],x[0][1],x[0][2],) for x in data["accel"]]
    gyros = [(
        x[0][0] * pi / 180,
        x[0][1] * pi / 180,
        x[0][2] * pi / 180,) for x in data["gyro"]]

    s = data["accel"][0][1]
    stamps = [x[1] - s for x in data["accel"]]

    # Find biases ------------------- 
    ax = np.mean([x[0] for x in accels])
    ay = np.mean([x[1] for x in accels])
    az = np.mean([x[2] for x in accels]) - 1.0
    print('Accels mean: {:.4f} {:.4f} {:.4f}'.format(ax,ay,az))

    gx = np.mean([x[0] for x in gyros])
    gy = np.mean([x[1] for x in gyros])
    gz = np.mean([x[2] for x in gyros])
    print('Gyro mean: {:.4f} {:.4f} {:.4f}'.format(gx,gy,gz))

    # Correct ----------------------
    accels = [(
        x[0] - ax,
        x[1] - ay,
        x[2] - az,) for x in accels]
    gyros = [(
        x[0] - gx,
        x[1] - gy,
        x[2] - gz,) for x in gyros]
    return accels, gyros, stamps


@app.cell
def _(accels, gyros, plt, stamps):
    for s_1, title in zip([accels, gyros], ["Acceration (g's)", 'Angular Rate (rads/sec)']):
        plt.figure()
        plt.plot(stamps, [x[0] for x in s_1], label='x')
        plt.plot(stamps, [x[1] for x in s_1], label='y')
        plt.plot(stamps, [x[2] for x in s_1], label='z')
        plt.grid(True)
        plt.title(title)
        plt.legend()
    return


@app.cell
def _(np, skew4):
    # System equations of motion (eom)
    # these follow the Titterton ECEF derivation
    def eom(t, X, u):
        """
        State vector
        X = tuple[vx vy vz px py pz qw qx qy qz]
        v - velocity
        p - position
        q - quaternion (orientation)

        These are sensor readings from IMU
        u = [fx fy fz wx wy wz]
        f - force (acceleration)
        w - angular velocity (from gyros)
        """
        # imu ---------------------
        f = 9.81*np.array(u[0:3]) 
        wx, wy, wz = u[3:]
        W = skew4(wx, wy, wz)

        # state -------------------
        v = np.array(X[0:3])
        p = np.array(X[3:6])
        q = np.array(X[6:])

        # nav const --------------
        wie = np.array([0, 0, 7.292115E-15])
        Ceb = np.eye(3) # ???
        g = np.array([0,0,-9.81])

        # update local gravity model
        gl = g - np.cross(wie, np.cross(wie, p))

        # velocity update
        vd = Ceb.dot(f)-2.0*np.cross(wie, np.cross(wie, v)) + gl

        # position update
        pd = v

        # orientation update
        qd = 0.5 * W.dot(q)

        # print('vd', vd)
        # print('pd', pd)
        # print('qd', qd)

        XX = np.hstack((vd, pd, qd))
        return XX

    return (eom,)


@app.cell
def _(Sim, accels, eom, gyros, stamps):
    s_2 = Sim(eom, accels, gyros, stamps)
    X, timestamps = s_2.run_nav()
    return X, timestamps


@app.cell
def _(X):
    vel = [x[:3] for x in X]
    pos = [x[3:6] for x in X]
    orient = [x[6:] for x in X]
    return orient, pos


@app.cell
def _(Quaternion, orient, plt, pos, timestamps):
    x_1 = [x[0] for x in pos]
    _y = [x[1] for x in pos]
    _z = [x[2] for x in pos]
    plt.figure()
    plt.plot(x_1, _y)
    plt.grid(True)
    plt.axis('equal')
    plt.title('Path')
    plt.ylabel('Y [m]')
    plt.xlabel('X [m]')
    plt.figure()
    plt.plot(timestamps, _z)
    plt.xlabel('Time (sec)')
    plt.grid(True)
    plt.title('Alt (m)')
    rpy = [Quaternion(*x).to_euler(degrees=True) for x in orient]
    r = [x[0] for x in rpy]
    p = [x[1] for x in rpy]
    _y = [x[2] for x in rpy]
    plt.figure()
    plt.plot(timestamps, r, label='Roll')
    plt.plot(timestamps, p, label='Pitch')
    plt.plot(timestamps, _y, label='Yaw')
    plt.grid(True)
    plt.xlabel('Time (sec)')
    plt.ylabel('Angle (deg)')
    plt.legend()
    plt.title('RPY')
    return (x_1,)


@app.cell
def _(Cnb, Cne, np, skew4):
    def eom2(t, X, u):
        """
        State vector
        X = tuple[vx vy vz px py pz qw qx qy qz]
        v - velocity
        p - position
        q - quaternion (orientation)

        These are sensor readings from IMU
        u = [fx fy fz wx wy wz]
        f - force (acceleration)
        w - angular velocity (from gyros)
        """
        # imu ---------------------
        f = 9.81*np.array(u[0:3]) 
        wx, wy, wz = u[3:]
        W = skew4(wx, wy, wz)

        # state -------------------
        v = np.array(X[0:3])
        p = np.array(X[3:6])
        q = np.array(X[6:])

        # nav const --------------
        wie = np.array([0, 0, 7.292115E-15])
        Ceb = Cne.dot(Cnb)
        g = np.array([0,0,-9.81])

        # update local gravity model
        gl = g - np.cross(wie, np.cross(wie, p))

        # velocity update
        vd = Ceb.dot(f)-2.0*np.cross(wie, np.cross(wie, v)) + gl

        # position update
        pd = v

        # orientation update
        qd = 0.5 * W.dot(q)

        # print('vd', vd)
        # print('pd', pd)
        # print('qd', qd)

        XX = np.hstack((vd, pd, qd))
        return XX

    return


@app.cell
def _(RK4, dot, inv, np):
    class EKF(object):
        """
        Extended Kalman Filter (EKF)

        def func(time, x, u):
            some nonlinear eqns
            return dx

        ekf = EKF(size_x,size_z)
        ekf.init(x, func, R, Q)

        while True:
            ekf.predict(u)
            x_hat = ekf.update(z)
        """
        def __init__(self, dim_x, dim_z, dt):
            self.dt = dt
            self.H = np.eye(dim_z)
            self.P = np.eye(dim_x)
            self.I = np.eye(dim_x)

        def init(self, x, f, r, q):
            """
            f: dx = f(x, u)
            r: measurement noise
            q: process noise
            """
            self.x = x
            self.rk = RK4(f)
            self.R = r
            self.Q = q
            self.time = 0.0

        def predict(self, u):
            # predict state estimate
            rk = self.rk
            dt = self.dt
            t = self.time
            x = self.x

            y = rk.step(x, u, t, dt)
            F = np.Jacobian(y)

            self.time = t+dt
            self.x = y

            # predict covariance estimate
            Q = self.Q
            P = self.P

            self.P = dot(F, dot(P, F.T)) + Q

        def update(self, z):
            H = self.H
            R = self.R
            I = self.I
            P = self.P
            x = self.x

            K = dot(P, dot(H.T, inv(dot(H, dot(P, H.T)) + R)))
            x = x + K.dot(z-H)
            p = (I - K.dot(H)).dot(P)

            self.x = x
            self.P = p

            return x

    return (EKF,)


@app.cell
def _(Sn, np, wn):
    def ins_corr(t,X,u):
        """
        Chatfield equations
        """
        wie = np.array([0,0,7.29211E-15])
        web = np.array([0,0,0])  # FIXME

        V = tuple(X[0:3])
        P = tuple(X[3:6])
        O = tuple(X[6:9])
        S = tuple(X[9:12])
        w = tuple(X[12:15])

        Reb = np.array([  # FIXME
            [1,0,0],
            [0,1,0],
            [0,0,1]
        ])

        Vd = -2*np.cross(wie,V) + np.cross(wie, np.cross(wie, P)) + np.cross(S, O) + Reb*S
        Pd = V
        Od = np.cross(web, O) - Reb*w
        Sd = Sn
        wd = wn

    return (ins_corr,)


@app.cell
def _(EKF, RK4, ins_corr, np, save, x_1):
    class Sim2(object):

        def __init__(self, a, g, tm):
            self.accels = a  # grab the inertial sensors
            self.gyros = g
            self.dts = tm

        def run_ins(self, nav, corr):
            """
            These are the navigation only equations, no EKF corrections
            """
            nav_rk = RK4(nav)
            ekf = EKF(15, 15, 0.05)
            ekf.init(x_1, ins_corr, 0.001 * np.eye(3), 0.001 * np.eye(3))
            vel = (0, 0, 0)
            pos = (0, 0, 0)
            orient = (1, 0, 0, 0)  # initial states
            X = np.array(vel + pos + orient)
            save_nav = []
            XX = np.zeros(15)
            last = self.dts[0] - 0.1
            for i, tm in enumerate(self.dts):
                u = self.accels[i] + self.gyros[i]
                X = nav_rk.step(X, u, tm, tm - last)
                last = tm
                save_nav.append(X)
                ekf
            print('>> Simulation end: {:.2f} seconds'.format(tm))
            print('>> Steps: {} at {:0.3f} sec'.format(len(save), np.mean(np.diff(self.dts))))
            return (save, self.dts)

    return (Sim2,)


@app.cell
def _(Sim2, accels, gyros, stamps):
    s_3 = Sim2(accels, gyros, stamps)
    return (s_3,)


@app.cell
def _(s_3):
    X_1, timestamps_1 = s_3.run_ins()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Discrete
    """)
    return


@app.cell
def _(atan2, np, pi, plt, sqrt):
    def discrete(accel, gyro):
        gc = np.mean([x[0][0] for x in gyro])
        axc = np.mean([x[0][0] for x in accel])
        ayc = np.mean([x[0][1] for x in accel])
        azc = np.mean([x[0][2] for x in accel])
        print('axc: {:.4f}  ayc: {:.4f}  azc: {:.4f}'.format(axc, ayc, azc))
        roll = 0
        told = gyro[0][1]
        tstart = told
        cfilter = 0.85
        save = []
        savea = []
        savet = []
        for (w, t), (a, t) in zip(gyro, accel):
            an = sqrt((a[0] - axc) ** 2 + (a[2] - azc + 1) ** 2)
            ax = atan2(a[1] - ayc, an) * 180 / pi  # y/z
            savea.append(ax)
            roll = roll + ((w[0] - gc) * (t - told) * cfilter + ax * (1 - cfilter))
            told = t  # complementary filter
            savet.append(t - tstart)
            save.append(roll)
        plt.plot(savet, save, label='gyro')
        plt.plot(savet, savea, label='accel')
        plt.legend()
        plt.grid(True)

    return (discrete,)


@app.cell
def _(accel, discrete, gyro):
    discrete(accel, gyro)
    return


@app.cell
def _(accel):
    accel[1][1] - accel[0][1]
    return


@app.cell
def _(accel, plt):
    plt.plot([x[0][2] for x in accel])
    return


if __name__ == "__main__":
    app.run()
