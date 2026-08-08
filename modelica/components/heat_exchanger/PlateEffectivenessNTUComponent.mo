model PlateEffectivenessNTUComponent
  parameter Real Q_flow_nominal(unit="W") = 1000000.0 annotation(Evaluate=false);
  parameter Real m1_flow_nominal(unit="kg/s") = 100.0 annotation(Evaluate=false);
  parameter Real m2_flow_nominal(unit="kg/s") = 120.0 annotation(Evaluate=false);
  parameter Real T_a1_nominal(unit="K") = 295.15 annotation(Evaluate=false);
  parameter Real T_b1_nominal(unit="K") = 290.15 annotation(Evaluate=false);
  parameter Real T_a2_nominal(unit="K") = 285.15 annotation(Evaluate=false);
  parameter Real T_b2_nominal(unit="K") = 290.15 annotation(Evaluate=false);
  parameter Real n1(min=0, max=1) = 0.8 annotation(Evaluate=false);
  parameter Real n2(min=0, max=1) = 0.8 annotation(Evaluate=false);
  parameter Real r_nominal(min=0) = 1.0 annotation(Evaluate=false);
  parameter Real cpWat(unit="J/(kg.K)") = 4186.0 annotation(Evaluate=false);

  Modelica.Blocks.Interfaces.RealInput T1In_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput T2In_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput m1_flow_kg_s(unit="kg/s");
  Modelica.Blocks.Interfaces.RealInput m2_flow_kg_s(unit="kg/s");

  Modelica.Blocks.Interfaces.RealOutput T1Out_C(unit="degC");
  Modelica.Blocks.Interfaces.RealOutput T2Out_C(unit="degC");
  Modelica.Blocks.Interfaces.RealOutput Q_flow_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput eps_s(unit="1");
  Modelica.Blocks.Interfaces.RealOutput UA_s(unit="W/K");

protected
  Real C1(unit="W/K");
  Real C2(unit="W/K");
  Real CMin(unit="W/K");
  Real CMax(unit="W/K");
  Real Cr(unit="1");
  Real NTU(unit="1");
  Real dT1_nominal(unit="K");
  Real dT2_nominal(unit="K");
  Real dTlm_nominal(unit="K");
  Real UA_nominal(unit="W/K");

equation
  C1 = max(0.0, m1_flow_kg_s) * cpWat;
  C2 = max(0.0, m2_flow_kg_s) * cpWat;
  CMin = min(C1, C2);
  CMax = max(C1, C2);
  Cr = CMin / max(1.0, CMax);
  dT1_nominal = max(0.1, T_a1_nominal - T_b2_nominal);
  dT2_nominal = max(0.1, T_b1_nominal - T_a2_nominal);
  dTlm_nominal = if abs(dT1_nominal - dT2_nominal) < 0.01 then dT1_nominal else (dT1_nominal - dT2_nominal) / log(dT1_nominal / dT2_nominal);
  UA_nominal = abs(Q_flow_nominal) / max(0.1, dTlm_nominal);
  UA_s = UA_nominal * (max(0.0, m1_flow_kg_s) / max(1e-6, m1_flow_nominal))^n1
    * (max(0.0, m2_flow_kg_s) / max(1e-6, m2_flow_nominal))^n2 * max(0.05, r_nominal);
  NTU = UA_s / max(1.0, CMin);
  eps_s = if abs(1.0 - Cr) < 0.001 then NTU / (1.0 + NTU) else (1.0 - exp(-NTU * (1.0 - Cr))) / (1.0 - Cr * exp(-NTU * (1.0 - Cr)));

  Q_flow_W = min(max(0.0, eps_s), 1.0) * CMin * max(0.0, T1In_C - T2In_C);
  T1Out_C = T1In_C - Q_flow_W / max(1.0, C1);
  T2Out_C = T2In_C + Q_flow_W / max(1.0, C2);

annotation(Documentation(info="<html><p>Signal-level plate effectiveness-NTU heat exchanger component for generated system models. Nominal heat flow and flow exponents are explicit calibrated parameters.</p></html>"));
end PlateEffectivenessNTUComponent;
