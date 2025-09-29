import pyomo.environ as pyo
import highspy as hp
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


plt.style.use("ggplot")


slvr = "appsi_highs"
print(slvr, "is Available:", pyo.SolverFactory(slvr).available(), "\n")


def build_concr_model(N, K, L, J_NL, alpha_coefficients, d_dates, p_times, sigma_availabilities):
  model_ = pyo.ConcreteModel()

  # Some model parameters related to index sets
  model_.N = pyo.Param(within=pyo.PositiveIntegers, initialize=N)
  model_.K = pyo.Param(within=pyo.PositiveIntegers, initialize=K)
  model_.L = pyo.Param(within=pyo.PositiveIntegers, initialize=L)

  # model index sets here, J_NL being the  subset of J
  model_.J = pyo.RangeSet(1, model_.N)
  model_.S = pyo.RangeSet(1, model_.K)
  model_.M = pyo.RangeSet(1, model_.L)

  model_.J_NL = pyo.Set(within=model_.J, initialize=J_NL)

  # Decision Variables which are all binary
  model_.x = pyo.Var(model_.S, model_.J, model_.M, within=pyo.Binary)

  # Actual model parameters related to the problem
  model_.alpha = pyo.Param(model_.J, within=pyo.PositiveReals, initialize=alpha_coefficients) # Importance Factors - coefficients
  model_.d = pyo.Param(model_.J, within=pyo.PositiveReals, initialize=d_dates) # Due dates for Not-Late jobs, although indexed for all j in J, only the ones for the j in J_NL will be used
  model_.p = pyo.Param(model_.J, model_.M, within=pyo.Binary, initialize=p_times) # Processing time for job j on processor/machine on m
  model_.sigma = pyo.Param(model_.S, model_.M, within=pyo.NonNegativeReals, initialize=sigma_availabilities) # Availability (can be > 1 if parallel machines exist) of processor-m on shift-s

  # Model Objective
  def z(_model):
    return sum(_model.alpha[_j] * sum(_s * _model.x[_s, _j, _model.L] for _s in _model.S) for _j in _model.J)

  model_.objective = pyo.Objective(rule=z, sense=pyo.minimize)

  # Constraints:
  # Enforcing due dates:
  def EnforceDueDates(_model, _j_NL):
    return sum(_s * _model.x[_s, _j_NL, _model.L] for _s in _model.S) <= _model.d[_j_NL]

  # Machining time constraint
  def MachineTimeConstraint(_model, _s, _m):
    return sum(_model.p[_j, _m] * _model.x[_s, _j, _m] for _j in _model.J) <= _model.sigma[_s, _m]

  # One Shift for One process Constraint
  def OneShiftConstraint(_model, _j, _m):
    return sum(_model.x[_s, _j, _m] for _s in _model.S) == 1

  # Sequantieal Machining Constraint
  def SequentialityConstraint(_model, _j, _m):
    if _m < _model.L:
      return _model.p[_j, _m] + sum(_s * (_model.x[_s, _j, _m] - _model.x[_s, _j, _m + 1]) for _s in _model.S) <= 0.0
    else:
      return pyo.Constraint.Skip

  model_.EnforceDueDates = pyo.Constraint(model_.J_NL, rule=EnforceDueDates)
  model_.MachineTimeLimit = pyo.Constraint(model_.S, model_.M, rule=MachineTimeConstraint)
  model_.OneShiftRule = pyo.Constraint(model_.J, model_.M, rule=OneShiftConstraint)
  model_.SequentialityConstraint = pyo.Constraint(model_.J, model_.M, rule=SequentialityConstraint)
  return model_


def prod_to_jobs(prod_df):
  jobs = {}
  N = len(prod_df.index)
  job_count = 0
  product_count = 0
  while N > 0:
    name = prod_df.index[product_count]
    a_product = prod_df.iloc[product_count]
    Qty, LS, DT, alpha = tuple(a_product)
    Qty, LS = int(Qty), int(LS)
    if Qty % LS == 0:
      k = Qty // LS
      for i in range(k):
        jobs[job_count] = (str(name) + f'_{i}', int(LS), int(DT), alpha, job_count) # for a job, in the strict technical sense,
        # processing time, due time, importance factor, and its assigned shift are required to be defined
        # name and part count are present for convenience
        job_count += 1
    elif Qty < LS:
      if Qty / LS >= 0.5 and alpha > 1.5:
        jobs[job_count] = (str(name), int(LS), int(DT), alpha, job_count)
      else:
        pass
    elif Qty > LS:
      if alpha >= 1.0:
        k = 1 + (Qty // LS)
        for i in range(k):
          jobs[job_count] = (str(name) + f'_{i}', int(LS), int(DT), alpha, job_count)
          job_count += 1
      else:
        k = Qty // LS
        for i in range(k):
          jobs[job_count] = (str(name) + f'_{i}', int(LS), int(DT), alpha, job_count)
          job_count += 1
    else:
      raise Exception("Unknown error.")

    N -= 1
    product_count += 1
  return pd.DataFrame(jobs, index=["OpName", "LotSize", "DueTime", "Imp.Factor", "AssignedShifts"]).transpose()


if __name__ == "__main__":
  gen = np.random.default_rng()
  test_product_number = 10
  # all test data are generated here randomly
  test_data = np.array([
      [gen.integers(low=1, high=19), gen.integers(low=1, high=16), gen.integers(low=1, high=26), gen.uniform(0.1, 5.0)] for _ in range(test_product_number)])

  # These numbers are used to 'name' the products, actual names of the products can take this place
  prod_numbers = np.arange(100, 1000)
  gen.shuffle(prod_numbers)
  prod = pd.DataFrame(test_data, columns=["Quantity", "LotSize", "DueTime", "ImportanceFactor"], index=[f'Part--{prod_numbers[i]}' for i in range(len(test_data))])
  print(prod)

  jobs_df = prod_to_jobs(prod)
  jobs_df.iloc[-10:, :]

  number_of_jobs = len(jobs_df.OpName)
  number_of_shifts = np.max(jobs_df.AssignedShifts)
  number_of_machines = 4

  importance_factors = jobs_df["Imp.Factor"].values
  due_dates = jobs_df["DueTime"].values
  proc_time = np.ones((number_of_jobs, number_of_machines,), dtype=int)
  available_machines = 2 * np.ones((number_of_shifts, number_of_machines,), dtype=int)
  due_enforced = tuple() # Empty set for now, so no due times are enforced

  Jobs2Plan_model = build_concr_model(number_of_jobs, number_of_shifts, number_of_machines,
                                      due_enforced,
                                      {i + 1 : importance_factors[i] for i in range(importance_factors.shape[0])},
                                      {i + 1 : due_dates[i] for i in range(due_dates.shape[0])},
                                      {(i + 1, j + 1) : proc_time[i, j] for i in range(proc_time.shape[0]) for j in range(proc_time.shape[1])},
                                      {(i + 1, j + 1) : available_machines[i, j] for i in range(available_machines.shape[0]) for j in range(available_machines.shape[1])}
                                      )

  opt = pyo.SolverFactory(slvr)
  resulting_plan = opt.solve(Jobs2Plan_model, tee=True)

  optimal_sol = Jobs2Plan_model.x.extract_values()

  for _j in range(number_of_jobs):
    assigned_shift = "("
    for _m in range(number_of_machines):
      s = sum([(_s + 1) * round(optimal_sol[(_s +  1, _j + 1, _m + 1)]) for _s in range(number_of_shifts)])
      assigned_shift += f"{s}, "
    assigned_shift += ")"
    jobs_df.loc[_j, "AssignedShifts"] = assigned_shift

  print("\n", jobs_df)
