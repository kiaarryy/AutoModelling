model MerkelComponent
  parameter Real m_flow_nominal(unit="kg/s") = 100.0 annotation(Evaluate=false);
  parameter Real ratWatAir_nominal(unit="1") = 1.67 annotation(Evaluate=false);
  parameter Real cWatFra[3] = {0.1082, 1.667, -0.7713} annotation(Evaluate=false);
  parameter Real yMin(min=0, max=1) = 0.05 annotation(Evaluate=false);
  parameter Real PFan_nominal(unit="W") = 10000.0 annotation(Evaluate=false);
  parameter Real fanRelPow_r_P[5] = {0.0, 0.0025, 0.0437, 0.265, 1.0} annotation(Evaluate=false);
  parameter Real cpWat(unit="J/(kg.K)") = 4180.0 annotation(Evaluate=false);

  Modelica.Blocks.Interfaces.RealInput Tin_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput Twb_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput m_flow_kg_s(unit="kg/s");
  Modelica.Blocks.Interfaces.RealInput y(unit="1");
  Modelica.Blocks.Interfaces.RealInput TRan_C(unit="K");

  Modelica.Blocks.Interfaces.RealOutput TOut_C(unit="degC");
  Modelica.Blocks.Interfaces.RealOutput Q_flow_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput PFan_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput TApp_C(unit="K");
  Modelica.Blocks.Interfaces.RealOutput FRWat(unit="1");
  Modelica.Blocks.Interfaces.RealOutput FRAir(unit="1");

protected
  Real uaRel(unit="1");
  Real fanRelPow(unit="1");

equation
  FRWat = max(0.0, m_flow_kg_s) / max(1e-6, m_flow_nominal);
  FRAir = min(max(y, yMin), 1.0);
  uaRel = max(0.05, cWatFra[1] + cWatFra[2] * FRWat + cWatFra[3] * FRWat^2);
  TApp_C = max(0.0, TRan_C / max(1.0, 1.0 + uaRel * FRAir / max(0.05, FRWat * ratWatAir_nominal)));

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

annotation(Documentation(info="<html><p>Signal-level Merkel cooling tower component for generated system models. It keeps calibrated Merkel UA correction and fan power parameters explicit and removes the calibration table driver.</p></html>"));
end MerkelComponent;
