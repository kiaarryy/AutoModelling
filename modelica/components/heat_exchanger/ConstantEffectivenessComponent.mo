model ConstantEffectivenessComponent
  parameter Real eps(min=0, max=1) = 0.8 annotation(Evaluate=false);
  parameter Real m1_flow_nominal(unit="kg/s") = 100.0 annotation(Evaluate=false);
  parameter Real m2_flow_nominal(unit="kg/s") = 120.0 annotation(Evaluate=false);
  parameter Real cpWat(unit="J/(kg.K)") = 4186.0 annotation(Evaluate=false);

  Modelica.Blocks.Interfaces.RealInput T1In_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput T2In_C(unit="degC");
  Modelica.Blocks.Interfaces.RealInput m1_flow_kg_s(unit="kg/s");
  Modelica.Blocks.Interfaces.RealInput m2_flow_kg_s(unit="kg/s");

  Modelica.Blocks.Interfaces.RealOutput T1Out_C(unit="degC");
  Modelica.Blocks.Interfaces.RealOutput T2Out_C(unit="degC");
  Modelica.Blocks.Interfaces.RealOutput Q_flow_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput eps_s(unit="1");

protected
  Real C1(unit="W/K");
  Real C2(unit="W/K");
  Real CMin(unit="W/K");

equation
  C1 = max(0.0, m1_flow_kg_s) * cpWat;
  C2 = max(0.0, m2_flow_kg_s) * cpWat;
  CMin = min(C1, C2);
  eps_s = min(max(eps, 0.0), 1.0);
  Q_flow_W = eps_s * CMin * max(0.0, T1In_C - T2In_C);
  T1Out_C = T1In_C - Q_flow_W / max(1.0, C1);
  T2Out_C = T2In_C + Q_flow_W / max(1.0, C2);

annotation(Documentation(info="<html><p>Signal-level constant-effectiveness water-water heat exchanger component for generated system models.</p></html>"));
end ConstantEffectivenessComponent;
