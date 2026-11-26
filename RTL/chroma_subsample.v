`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 26.11.2026 19:45:09
// Design Name: 
// Module Name: chroma_subsample
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


module chroma_subsample #(
    parameter IMG_W = 1920
) (
    input  wire clk,
    input  wire rst,
    input  wire in_valid,
    input  wire [7:0] in_y,
    input  wire [7:0] in_cb,
    input  wire [7:0] in_cr,
    output reg out_y_valid,
    output reg [7:0] out_y,
    output reg out_c_valid,
    output reg [7:0] out_cb,
    output reg [7:0] out_cr
);
    localparam WC     = IMG_W / 2;
    localparam COL_W  = $clog2(IMG_W);
    localparam PAIR_W = $clog2(WC);
    reg [COL_W-1:0] col;
    reg row_odd;
    reg [7:0] cb_left, cr_left;
    reg [17:0] top_pair;

    wire [PAIR_W-1:0] pair_idx = col[COL_W-1:1];
    wire [8:0] cb_pair = {1'b0, cb_left} + {1'b0, in_cb};
    wire [8:0] cr_pair = {1'b0, cr_left} + {1'b0, in_cr};
    wire [9:0] cb_quad = {1'b0, top_pair[17:9]} + {1'b0, cb_pair} + 10'd2;
    wire [9:0] cr_quad = {1'b0, top_pair[8:0]}  + {1'b0, cr_pair} + 10'd2;

    (* ram_style = "block" *) reg [17:0] lb [0:WC-1];
    
    always @(posedge clk) begin
    if (rst) begin
        col <= {COL_W{1'b0}};
        row_odd     <= 1'b0;
        out_y_valid <= 1'b0;
        out_c_valid <= 1'b0;
    end 
    else begin 
        out_y_valid <= in_valid;
        out_y  <= in_y;
        out_c_valid <= 1'b0;
        if (in_valid) begin
            if (!col[0]) begin
                cb_left <=in_cb;
                cr_left<=in_cr;
                if(row_odd) begin
                    top_pair <= lb[pair_idx];
                end    
            end
            else begin 
                if (!row_odd) begin
                    lb[pair_idx] <= {cb_pair, cr_pair};
                end 
                else begin
                    out_cb      <= cb_quad[9:2];
                    out_cr      <= cr_quad[9:2];
                    out_c_valid <= 1'b1;
                end
                
            end
            if (col == IMG_W - 1) begin
                col     <= 0;
                row_odd <= ~row_odd;
            end else begin
                col     <= col + 1'b1;
            end
       end
    end
end
    
endmodule
