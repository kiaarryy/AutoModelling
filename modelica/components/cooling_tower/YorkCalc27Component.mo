model YorkCalc27Component
  parameter Real m_flow_nominal(unit="kg/s") = 100.0 annotation(Evaluate=false);
  parameter Real yMin(min=0, max=1) = 0.05 annotation(Evaluate=false);
  parameter Real RlgMin(min=0) = 0.05 annotation(Evaluate=false);
  parameter Real RlgMax(min=0) = 8.0 annotation(Evaluate=false);
  parameter Real TAppMin_C = 0.0 annotation(Evaluate=false);
  parameter Real TAppMax_C = 40.0 annotation(Evaluate=false);
  parameter Real cpWat(unit="J/(kg.K)") = 4180.0 annotation(Evaluate=false);
  parameter Real PFan_nominal(unit="W") = 10000.0 annotation(Evaluate=false);
  parameter Real fanRelPow_r_P[5] = {0.0, 0.0025, 0.0437, 0.265, 1.0} annotation(Evaluate=false);
  parameter Real f[27] = {
    -0.359741205,
    -0.055053608,
    0.0023850432,
    0.173926877,
    -0.0248473764,
    0.00048430224,
    -0.005589849456,
    0.0005770079712,
    -0.00001342427256,
    2.84765801111111,
    -0.121765149,
    0.0014599242,
    1.680428651,
    -0.0166920786,
    -0.0007190532,
    -0.025485194448,
    0.0000487491696,
    0.00002719234152,
    -0.0653766255555556,
    -0.002278167,
    0.0002500254,
    -0.0910565458,
    0.00318176316,
    0.000038621772,
    -0.0034285382352,
    0.00000856589904,
    -0.000001516821552} annotation(Evaluate=false);

  Modelica.Blocks.Interfaces.RealInput Tin_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput Twb_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput m_flow_kg_s(unit="kg/s");
  Modelica.Blocks.Interfaces.RealInput y(unit="1");
  Modelica.Blocks.Interfaces.RealInput TRan_C(unit="K");

  Modelica.Blocks.Interfaces.RealOutput TOut_C(unit="degC");
  Modelica.Blocks.Interfaces.RealOutput Q_flow_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput PFan_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput TApp_C(unit="K");
  Modelica.Blocks.Interfaces.RealOutput Rlg(unit="1");

protected
  Real FRWat(unit="1");
  Real FRAir(unit="1");
  Real TApp_raw_C(unit="K");
  Real fanRelPow(unit="1");

equation
  FRWat = max(0.0, m_flow_kg_s) / max(1e-6, m_flow_nominal);
  FRAir = min(max(y, yMin), 1.0);
  Rlg = min(max(FRWat / max(1e-4, FRAir), RlgMin), RlgMax);

  TApp_raw_C =
      f[1]
    + f[2] * Twb_C
    + f[3] * Twb_C^2
    + f[4] * TRan_C
    + f[5] * Twb_C * TRan_C
    + f[6] * Twb_C^2 * TRan_C
    + f[7] * TRan_C^2
    + f[8] * Twb_C * TRan_C^2
    + f[9] * Twb_C^2 * TRan_C^2
    + f[10] * Rlg
    + f[11] * Twb_C * Rlg
    + f[12] * Twb_C^2 * Rlg
    + f[13] * TRan_C * Rlg
    + f[14] * Twb_C * TRan_C * Rlg
    + f[15] * Twb_C^2 * TRan_C * Rlg
    + f[16] * TRan_C^2 * Rlg
    + f[17] * Twb_C * TRan_C^2 * Rlg
    + f[18] * Twb_C^2 * TRan_C^2 * Rlg
    + f[19] * Rlg^2
    + f[20] * Twb_C * Rlg^2
    + f[21] * Twb_C^2 * Rlg^2
    + f[22] * TRan_C * Rlg^2
    + f[23] * Twb_C * TRan_C * Rlg^2
    + f[24] * Twb_C^2 * TRan_C * Rlg^2
    + f[25] * TRan_C^2 * Rlg^2
    + f[26] * Twb_C * TRan_C^2 * Rlg^2
    + f[27] * Twb_C^2 * TRan_C^2 * Rlg^2;
  TApp_C = min(max(TApp_raw_C, TAppMin_C), TAppMax_C);

  fanRelPow = if FRAir <= 0.1 then
      fanRelPow_r_P[1] + (fanRelPow_r_P[2] - fanRelPow_r_P[1]) * FRAir / 0.1
    elseif FRAir <= 0.3 then
      fanRelPow_r_P[2] + (fanRelPow_r_P[3] - fanRelPow_r_P[2]) * (FRAir - 0.1) / 0.2
    elseif FRAir <= 0.6 then
      fanRelPow_r_P[3] + (fanRelPow_r_P[4] - fanRelPow_r_P[3]) * (FRAir - 0.3) / 0.3
    else
      fanRelPow_r_P[4] + (fanRelPow_r_P[5] - fanRelPow_r_P[4]) * (FRAir - 0.6) / 0.4;

  TOut_C = Twb_C + TApp_C;
  Q_flow_W = max(0.0, m_flow_kg_s) * cpWat * (Tin_C - TOut_C);
  PFan_W = PFan_nominal * fanRelPow;

annotation(
  Documentation(info="<html><p>Signal-level YorkCalc 27-coefficient cooling tower component for system assembly. It exposes explicit operating-point inputs plus T/Q/P outputs.</p></html>"));
end YorkCalc27Component;
