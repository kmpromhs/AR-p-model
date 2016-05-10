  from nipy.algorithms.statistics.api import Term, Formula
  data = np.rec.fromarrays(([1,3,4,5,8,10,9], range(1,8)),
    ...                          names=('Y', 'X'))
  f = Formula([Term("X"), 1])
  dmtx = f.design(data, return_float=True)
  model = ARModel(dmtx, 2)
    We go through the ``model.iterative_fit`` procedure long-hand:
    for i in range(6):
    ...     results = model.fit(data['Y'])
    ...     print("AR coefficients:", model.rho)
    ...     rho, sigma = yule_walker(data["Y"] - results.predicted,
    ...                              order=2,
    ...                              df=model.df_resid)
    ...     model = ARModel(model.design, rho) #doctest: +FP_6DP
    ...
    AR coefficients: [ 0.  0.]
    AR coefficients: [-0.61530877 -1.01542645]
    AR coefficients: [-0.72660832 -1.06201457]
    AR coefficients: [-0.7220361  -1.05365352]
    AR coefficients: [-0.72229201 -1.05408193]
    AR coefficients: [-0.722278   -1.05405838]
  results.theta #doctest: +FP_6DP
    array([ 1.59564228, -0.58562172])
  results.t() #doctest: +FP_6DP
    array([ 38.0890515 ,  -3.45429252])
  print(results.Tcontrast([0,1]))  #doctest: +FP_6DP
    <T contrast: effect=-0.58562172384377043, sd=0.16953449108110835,
    t=-3.4542925165805847, df_den=5>
  print(results.Fcontrast(np.identity(2)))  #doctest: +FP_6DP
    <F contrast: F=4216.810299725842, df_den=5, df_num=2>
    Reinitialize the model, and do the automated iterative fit
  model.rho = np.array([0,0])
  model.iterative_fit(data['Y'], niter=3)
  print(model.rho)  #doctest: +FP_6DP
    [-0.7220361  -1.05365352]
    """
