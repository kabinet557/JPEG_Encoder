`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 26.11.2026 16:05:13
// Design Name: 
// Module Name: rgb_to_ycbcr
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////

module rgb2ycbcr #(
    parameter USER_W = 1
) (
    input wire clk,
    input wire rst,
    input wire in_valid,
    input wire [7:0] in_r,
    input wire [7:0] in_g,
    input wire [7:0] in_b,
    input wire [USER_W-1:0] in_user,
    output wire out_valid,
    output wire [7:0] out_y,
    output wire [7:0] out_cb,
    output wire [7:0] out_cr,
    output wire [USER_W-1:0] out_user
);

    localparam signed [17:0] C_YR  =  18'sd19595;
    localparam signed [17:0] C_YG  =  18'sd38470;
    localparam signed [17:0] C_YB  =  18'sd7471;
    localparam signed [17:0] C_CBR = -18'sd11059;
    localparam signed [17:0] C_CBG = -18'sd21709;
    localparam signed [17:0] C_CBB =  18'sd32768;
    localparam signed [17:0] C_CRR =  18'sd32768;
    localparam signed [17:0] C_CRG = -18'sd27439;
    localparam signed [17:0] C_CRB = -18'sd5329;
    localparam signed [27:0] ROUND =  28'sd32768;
    
    wire signed [8:0] r_s = {1'b0, in_r};
    wire signed [8:0] g_s = {1'b0, in_g};
    wire signed [8:0] b_s = {1'b0, in_b};
    
    reg [USER_W-1:0] u1, u2, u3;
    reg [2:0]        v_pipe;
    
    reg signed [26:0] p_yr,  p_yg,  p_yb;
    reg signed [26:0] p_cbr, p_cbg, p_cbb;
    reg signed [26:0] p_crr, p_crg, p_crb;
    
    reg signed [27:0] y_sum,cb_sum,cr_sum;
    
    wire signed [11:0] y_s  = y_sum  >>> 16;
    wire signed [11:0] cb_s = (cb_sum >>> 16) + 12'sd128;
    wire signed [11:0] cr_s = (cr_sum >>> 16) + 12'sd128;
    
    reg [7:0] y_q, cb_q, cr_q;
    
    always @(posedge clk) begin
        p_yr <= C_YR*r_s;
        p_yg <= C_YG*g_s;
        p_yb <= C_YB * b_s;
        p_cbr <= C_CBR*r_s;
        p_cbg <= C_CBG*g_s;
        p_cbb <= C_CBB*b_s;
        p_crr <= C_CRR*r_s;
        p_crg <= C_CRG*g_s;
        p_crb <= C_CRB*b_s;
        u1    <= in_user;
  
        y_sum  <= p_yr  + p_yg  + p_yb  + ROUND;
        cb_sum <= p_cbr + p_cbg + p_cbb + ROUND;
        cr_sum <= p_crr + p_crg + p_crb + ROUND;
        u2     <= u1;
        
        y_q  <= y_s[7:0];
        cb_q <= (cb_s < 0) ? 8'd0 : (cb_s > 255) ? 8'd255 : cb_s[7:0];
        cr_q <= (cr_s < 0) ? 8'd0 : (cr_s > 255) ? 8'd255 : cr_s[7:0];
        u3   <= u2;
        
        end
         always @(posedge clk) begin
        if (rst)
            v_pipe <= 3'b000;
        else     
            v_pipe <= {v_pipe[1:0], in_valid};
        end

    assign out_valid = v_pipe[2];
    assign out_y     = y_q;
    assign out_cb    = cb_q;
    assign out_cr    = cr_q;
    assign out_user  = u3;
    
endmodule
