########################
#### PROCESSORS   ######
########################
def remove_trend_poly(planetfilede, data, poly_d=10):
    data,label = data
    def fun(c, *argos, **kwargs):
        return sum([c*args[0]**e for e,c in zip(c, range(poly_d))])
    # Fit curve
    model = make_pipeline(PolynomialFeatures(poly_d), RANSACRegressor())
    indicies = np.array(range(len(data)))
    try:
        model.fit(indicies, data)
    except:
        # Couldn't fit
        return data, label
    trend = model.predict(indicies)
    return data-trend, label


def split_periods(t0, flux, times, period):
    # Use a quantization method
    # Split periods
    split_periods = []
    split_fluxes = []
    i = 0
    centered_times = (times-t0-period*.5)%period
    while i < len(centered_times)-1:
        c_periods = []
        c_fluxes = []
        while i < len(centered_times)-1 and centered_times[i+1] > centered_times[i]:
            c_periods.append(centered_times[i])
            c_fluxes.append(flux[i])
            i+=1
        i+=1
        split_periods.append(c_periods)
        split_fluxes.append(c_fluxes)
    return split_periods, split_fluxes

def interpolate_to_grid(data, resolution=10000):
    # Create spline and interpolate for each.
    interpolated_fluxes = []
    resolution = 10000
    new_indicies = np.linspace(0, period, resolution)
    num_periods = len(split_periods)
    counter = np.zeros((num_periods,resolution))+num_periods
    for i in range(len(split_periods)):
        spl = UnivariateSpline(split_periods[i], split_fluxes[i], k=5,s=0.000001)
        indicies = 0
        interperted = spl(new_indicies)
        min_val = min(split_periods[i])
        max_val = max(split_periods[i])
        i2 = 0
        while i2 < len(interperted):
            if new_indicies[i2] < min_val:
                interperted[i2] = 0
                counter[i][i2] -= 1
            i2+=1
        i2 = len(interperted)-1
        while i2 >= 0:
            if new_indicies[i2] > max_val:
                interperted[i2] = 0
                counter[i][i2] -= 1
            i2-=1
        interpolated_fluxes.append(interperted)
    interpolated_fluxes = np.array(interpolated_fluxes)

def split_all_above_thresh(snr_thresh=8):
    planetfilede.get_bls_data()
    snr = planetfilede.snr
    snr_periods = planetfilede.snr_periods
    # Gets all the periods with an snr greater than snr_thresh
    snr_periods_filtered = [x for i, v in enumerate([snr > snr_thresh]) if v]
    samples = []
    for period in snr_periods:
        sample = split_periods(planetfilede.t0, planetfilede.flux, planetfilede.timestamps)

        samples.append()




def sum_along_period(planetfilede, data, normalize=True, resolution=1000):
    split_periods, split_fluxes, label = data
    interpolated_fluxes = []
    resolution = 1000
    new_indicies = np.linspace(0, period, resolution)
    num_periods = len(split_periods)
    counter = np.zeros((num_periods,resolution))+num_periods
    for i in range(len(split_periods)):
        spl = UnivariateSpline(split_periods[i], split_fluxes[i], k=5,s=0.000001)
        indicies = 0
        interperted = spl(new_indicies)
        min_val = min(split_periods[i])
        max_val = max(split_periods[i])
        i2 = 0
        while i2 < len(interperted):
            if new_indicies[i2] < min_val:
                interperted[i2] = 0
                counter[i][i2] -= 1
            i2+=1
        i2 = len(interperted)-1
        while i2 >= 0:
            if new_indicies[i2] > max_val:
                interperted[i2] = 0
                counter[i][i2] -= 1
            i2-=1
        interpolated_fluxes.append(interperted)
    if normalize:
        interpolated_fluxes = np.array([x/max(abs(x)) for x in interpolated_fluxes])
    return np.array(interpolated_fluxes), label

