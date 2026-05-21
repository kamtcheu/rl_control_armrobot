% 
%
%   Computation of the trajectory and orientation for an elipse
%
%       x-side = a
%       y-side = b
%       Using a parametric equation as a function of phi (angle)
%
%   2024-06-07

clear all
clf

a = 1.5;
b = 1.0;
k = (a^2 - b^2) / a^2;
Rn = 12;
theta_h = + 90 / 180 * pi;

elipse_center_global = [(- 289.997) (+ 6.666) (+ 247.185)]';

R_global2base = [0 1 0;
                 0 0 1;
                 1 0 0];
d_global2base = 0 * [(+ 30) (+ 30) (+ 30)]';
T_global2base = R_global2base;
T_global2base(:, 4) = d_global2base;
T_global2base(4, :) = [0 0 0 1];

R_base2global = R_global2base';
d_base2global = - R_base2global * d_global2base;
T_base2global = R_base2global;
T_base2global(:, 4) = d_base2global;
T_base2global(4, :) = [0 0 0 1];

R_tcp2t = [1   0                            0;
           0   (cos(theta_h - pi / 2))     (- sin(theta_h - pi / 2));
           0   (sin(theta_h - pi / 2))     (cos(theta_h - pi / 2))];
d_tcp2t = [0 (- Rn * sin(theta_h)) (Rn * (cos(theta_h) - 1))]';
T_tcp2t = R_tcp2t;
T_tcp2t(:, 4) = d_tcp2t;
T_tcp2t(4, :) = [0 0 0 1];

R_t2tcp = R_tcp2t';
d_t2tcp = - R_tcp2t * d_tcp2t;
T_t2tcp = R_t2tcp;
T_t2tcp(:, 4) = d_t2tcp;
T_t2tcp(4, :) = [0 0 0 1];

R_x_m90 = [1 0 0;
            0 cos(- pi / 2) (- sin(- pi / 2));
            0 sin(- pi / 2) cos(- pi / 2)];

phi_deg = 180 : 10 : 360;
phi = phi_deg * pi / 180;
delta_z = 0.0;

for i = 1:1:19
    P_r(i) = sqrt(b^2 / (1 - k * (cos(phi(i)))^2));
    P_t(:, i) = [0.0 (P_r(i) * cos(phi(i))) (P_r(i) * sin(phi(i)))]' + elipse_center_global + [0 0 delta_z]';

    P_r_dot(i) = (-0.5 * b * sin(2 * phi(i)) * k) / (1 - k * (cos(phi(i)))^2)^1.5;
    O_t(:, i) = [0;
                    (- P_r(i) * sin(phi(i)) + P_r_dot(i) * cos(phi(i)));
                    (+ P_r(i) * cos(phi(i)) + P_r_dot(i) * sin(phi(i)))];
    O_mag = norm(O_t(:, i));
    O_t(:, i) = O_t(:, i) / O_mag;
    
    P_t_homogen = [P_t(:, i)' 1]';
    P_tcp_homogen = T_t2tcp * P_t_homogen;
    P_tcp(:, i) = P_tcp_homogen(1:3);
    O_tcp(:, i) = R_x_m90 * R_t2tcp * O_t(:, i);

    P_base_homogen = T_global2base * P_tcp_homogen;
    P_base(:, i) = P_base_homogen(1:3);
    O_base(:, i) = R_global2base * O_tcp(:, i);
    
    if i > 10
        delta_z = delta_z + 0* 0.2;
    end

end  

figure(1)
    subplot(1,3,1)
        plot3(P_t(1, :), P_t(2, :), P_t(3, :), 'b', "linewidth", 2)
        title('Target Needle Tip (global position + global orientation)')
        hold on
        quiver3(P_t(1, :), P_t(2, :), P_t(3, :), O_t(1, :), O_t(2, :), O_t(3, :), "r")
        hold off
        axis equal
        view(90, 0)
        xlabel('x_t')
        ylabel('y_t')
        zlabel('z_t')
        grid
        
    subplot(1,3,2)
        plot3(P_tcp(1, :), P_tcp(2, :), P_tcp(3, :), 'b', "linewidth", 2)
        title('Target tcp (global position')
        hold on
        quiver3(P_tcp(1, :), P_tcp(2, :), P_tcp(3, :), O_tcp(1, :), O_tcp(2, :), O_tcp(3, :), "r")
        hold off
        axis equal
%        view(90, 0)
        xlabel('x_t_c_p')
        ylabel('y_t_c_p')
        zlabel('z_t_c_p')
        grid
        
    subplot(1,3,3)
        plot3(P_base(1, :), P_base(2, :), P_base(3, :), 'b', "linewidth", 2)
        title('Required Base position and orientation')
        hold on
        quiver3(P_base(1, :), P_base(2, :), P_base(3, :), O_base(1, :), O_base(2, :), O_base(3, :), "r")
        hold off
        axis equal
%         view(0, 90)
        xlabel('x_b_a_s_e')
        ylabel('y_b_a_s_e')
        zlabel('z_b_a_s_e')
        grid