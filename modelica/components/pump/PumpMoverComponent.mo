model PumpMoverComponent
  parameter Real m_flow_nominal(unit="kg/s") = 100.0 annotation(Evaluate=false);
  parameter Real dp_nominal(unit="Pa") = 300000.0 annotation(Evaluate=false);
  parameter Real dp_system_nominal(unit="Pa") = 250000.0 annotation(Evaluate=false);
  parameter Real P_nominal(unit="W") = 40000.0 annotation(Evaluate=false);
  parameter Real P_scale(unit="1") = 1.0 annotation(Evaluate=false);
  parameter Real y_min(unit="1") = 0.05 annotation(Evaluate=false);
  parameter Real y_max(unit="1") = 1.20 annotation(Evaluate=false);
  parameter Real rho_nominal(unit="kg/m3") = 997.0 annotation(Evaluate=false);
  parameter Real g(unit="m/s2") = 9.80665 annotation(Evaluate=false);

  Modelica.Blocks.Interfaces.RealInput y(unit="1");

  Modelica.Blocks.Interfaces.RealOutput P_W(unit="W");
  Modelica.Blocks.Interfaces.RealOutput m_flow_s(unit="kg/s");
  Modelica.Blocks.Interfaces.RealOutput dp_s(unit="Pa");
  Modelica.Blocks.Interfaces.RealOutput head_s(unit="m");
  Modelica.Blocks.Interfaces.RealOutput y_s(unit="1");

protected
  Real flowRatio(unit="1");

equation
  y_s = min(max(y, y_min), y_max);
  flowRatio = sqrt(max(0.0, y_s^2 * dp_nominal / max(1.0, dp_system_nominal)));
  m_flow_s = m_flow_nominal * flowRatio;
  dp_s = dp_system_nominal * flowRatio^2;
  head_s = dp_s / max(1.0, rho_nominal * g);
  P_W = max(0.0, P_scale * P_nominal * y_s^3 * max(0.0, flowRatio));

annotation(Documentation(info="<html><p>Signal-level pump mover component for generated system models. It approximates the Buildings SpeedControlled_y plus quadratic system resistance candidate using explicit calibrated parameters.</p></html>"));
end PumpMoverComponent;
