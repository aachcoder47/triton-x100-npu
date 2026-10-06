// ============================================================================
// Module: tt_um_triton_tile_npu
// Description: Tiny Tapeout Top-Level Wrapper for Triton-X100 Tile NPU Core.
//              Integrates 2D Systolic Matrix Engine and Asynchronous Tile DMA.
// ============================================================================

`timescale 1ns / 1ps

module tt_um_triton_tile_npu (
    input  wire [7:0] ui_in,    // Dedicated inputs: [7]=compute_en, [6]=clear_acc, [5:3]=B_in, [2:0]=A_in
    output wire [7:0] uo_out,   // Dedicated outputs: Multiplexed Matrix Accumulator Byte
    input  wire [7:0] uio_in,   // Bidirectional inputs: [0]=dma_start, [3:1]=stride, [7:4]=byte_sel
    output wire [7:0] uio_out,  // Bidirectional outputs: [0]=dma_busy, [1]=dma_done, [2]=mem_req_valid, [7:3]=addr_lsb
    output wire [7:0] uio_oe,   // Bidirectional output enables (0=input, 1=output)
    input  wire       ena,      // Will go high when design is selected
    input  wire       clk,      // System clock
    input  wire       rst_n     // Active-low reset
);

    // Control signals
    wire compute_en = ui_in[7] & ena;
    wire clear_acc  = ui_in[6];
    wire [1:0] byte_select = uio_in[5:4];

    // Systolic Matrix inputs (packed from 8-bit bus)
    wire [31:0] a_row_in = {8'b0, 8'b0, {5'b0, ui_in[2:0]}, {5'b0, ui_in[2:0]}};
    wire [31:0] b_col_in = {8'b0, 8'b0, {5'b0, ui_in[5:3]}, {5'b0, ui_in[5:3]}};
    wire [511:0] c_tile_out;

    // DMA control
    wire dma_start = uio_in[0] & ena;
    wire [31:0] dma_base_ptr = 32'h1000_0000;
    wire [31:0] dma_stride_m = {29'b0, uio_in[3:1]};
    wire dma_busy;
    wire dma_done;
    wire mem_req_valid;
    wire [31:0] mem_req_addr;

    // Instantiate Top-level TEU Core
    triton_teu_top #(
        .DATA_WIDTH(8),
        .ACC_WIDTH(32),
        .TILE_SIZE(4),
        .ADDR_WIDTH(32)
    ) teu_core (
        .clk(clk),
        .rst_n(rst_n),
        .compute_en(compute_en),
        .clear_acc(clear_acc),
        .a_row_in(a_row_in),
        .b_col_in(b_col_in),
        .c_tile_out(c_tile_out),
        .dma_start(dma_start),
        .dma_base_ptr(dma_base_ptr),
        .dma_block_m(16'd4),
        .dma_block_k(16'd4),
        .dma_stride_m(dma_stride_m),
        .dma_busy(dma_busy),
        .dma_done(dma_done),
        .mem_req_valid(mem_req_valid),
        .mem_req_addr(mem_req_addr),
        .mem_req_ready(1'b1)
    );

    // Multiplex 32-bit PE accumulator output onto 8-bit uo_out bus
    reg [7:0] out_mux;
    always @(*) begin
        case (byte_select)
            2'b00: out_mux = c_tile_out[7:0];
            2'b01: out_mux = c_tile_out[15:8];
            2'b10: out_mux = c_tile_out[23:16];
            2'b11: out_mux = c_tile_out[31:24];
        endcase
    end
    assign uo_out = out_mux;

    // Bidirectional Pin Assignments:
    // uio[3:0] are inputs (dma_start, stride) -> OE = 0
    // uio[7:4] are outputs (dma_busy, dma_done, mem_req_valid, addr) -> OE = 1
    assign uio_oe = 8'b1111_0000;
    assign uio_out[4] = dma_busy;
    assign uio_out[5] = dma_done;
    assign uio_out[6] = mem_req_valid;
    assign uio_out[7] = mem_req_addr[2];
    assign uio_out[3:0] = 4'b0000;

endmodule
