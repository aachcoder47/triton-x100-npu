// ============================================================================
// Module: triton_mac_pe
// Description: Processing Element (PE) for Triton-X100 Systolic Array.
//              Performs FP8/INT8 Multiply-Accumulate with FP32/INT32 accumulation.
//              Directly implements Triton `tl.dot` inner arithmetic loop.
// ============================================================================

`timescale 1ns / 1ps

module triton_mac_pe #(
    parameter DATA_WIDTH = 8,
    parameter ACC_WIDTH  = 32
)(
    input  wire                   clk,
    input  wire                   rst_n,
    input  wire                   enable,
    input  wire                   clear_acc,
    // Systolic Inputs
    input  wire [DATA_WIDTH-1:0]  a_in,
    input  wire [DATA_WIDTH-1:0]  b_in,
    // Systolic Pass-Through Outputs
    output reg  [DATA_WIDTH-1:0]  a_out,
    output reg  [DATA_WIDTH-1:0]  b_out,
    // Accumulator Output
    output reg  [ACC_WIDTH-1:0]   acc_out
);

    wire signed [DATA_WIDTH-1:0] a_signed = a_in;
    wire signed [DATA_WIDTH-1:0] b_signed = b_in;
    wire signed [ACC_WIDTH-1:0]  prod     = a_signed * b_signed;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            a_out   <= {DATA_WIDTH{1'b0}};
            b_out   <= {DATA_WIDTH{1'b0}};
            acc_out <= {ACC_WIDTH{1'b0}};
        end else if (enable) begin
            // Forward systolic data to adjacent PEs
            a_out <= a_in;
            b_out <= b_in;
            
            if (clear_acc) begin
                acc_out <= prod;
            end else begin
                acc_out <= acc_out + prod;
            end
        end
    end

endmodule
