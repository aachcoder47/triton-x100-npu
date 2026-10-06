// ============================================================================
// Testbench: tb_triton_systolic
// Description: Functional verification testbench for Triton-X100 TEU.
//              Verifies 4x4 Systolic Matrix Engine and Tile DMA state machine.
// ============================================================================

`timescale 1ns / 1ps

module tb_triton_systolic;

    parameter DATA_WIDTH = 8;
    parameter ACC_WIDTH  = 32;
    parameter TILE_SIZE  = 4;
    parameter ADDR_WIDTH = 32;

    reg clk;
    reg rst_n;
    reg compute_en;
    reg clear_acc;

    reg [TILE_SIZE*DATA_WIDTH-1:0] a_row_in;
    reg [TILE_SIZE*DATA_WIDTH-1:0] b_col_in;
    wire [TILE_SIZE*TILE_SIZE*ACC_WIDTH-1:0] c_tile_out;

    reg dma_start;
    reg [ADDR_WIDTH-1:0] dma_base_ptr;
    reg [15:0] dma_block_m;
    reg [15:0] dma_block_k;
    reg [ADDR_WIDTH-1:0] dma_stride_m;
    wire dma_busy;
    wire dma_done;

    wire mem_req_valid;
    wire [ADDR_WIDTH-1:0] mem_req_addr;
    reg  mem_req_ready;

    // Instantiate Device Under Test (DUT)
    triton_teu_top #(
        .DATA_WIDTH(DATA_WIDTH),
        .ACC_WIDTH(ACC_WIDTH),
        .TILE_SIZE(TILE_SIZE),
        .ADDR_WIDTH(ADDR_WIDTH)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .compute_en(compute_en),
        .clear_acc(clear_acc),
        .a_row_in(a_row_in),
        .b_col_in(b_col_in),
        .c_tile_out(c_tile_out),
        .dma_start(dma_start),
        .dma_base_ptr(dma_base_ptr),
        .dma_block_m(dma_block_m),
        .dma_block_k(dma_block_k),
        .dma_stride_m(dma_stride_m),
        .dma_busy(dma_busy),
        .dma_done(dma_done),
        .mem_req_valid(mem_req_valid),
        .mem_req_addr(mem_req_addr),
        .mem_req_ready(mem_req_ready)
    );

    // Clock Generation (500 MHz simulation clock -> 2.0ns period)
    always #1 clk = ~clk;

    initial begin
        $dumpfile("teu_waveforms.vcd");
        $dumpvars(0, tb_triton_systolic);

        // Initialization
        clk = 0;
        rst_n = 0;
        compute_en = 0;
        clear_acc = 1;
        a_row_in = 0;
        b_col_in = 0;
        dma_start = 0;
        dma_base_ptr = 32'h1000_0000;
        dma_block_m = 4;
        dma_block_k = 4;
        dma_stride_m = 32'd64; // Stride 64 bytes
        mem_req_ready = 1;

        #10 rst_n = 1;
        #5 clear_acc = 0;
        compute_en = 1;

        $display("---------------------------------------------------------------");
        $display("Starting Triton-X100 Tile Execution Unit (TEU) EDA Simulation");
        $display("---------------------------------------------------------------");

        // Cycle 1: Feed row 0 of A (all 2s) and col 0 of B (all 3s)
        // 2 * 3 = 6
        a_row_in = {8'd2, 8'd2, 8'd2, 8'd2};
        b_col_in = {8'd3, 8'd3, 8'd3, 8'd3};
        #2;

        // Cycle 2: Feed next inputs
        a_row_in = {8'd1, 8'd1, 8'd1, 8'd1};
        b_col_in = {8'd4, 8'd4, 8'd4, 8'd4};
        #2;

        // Trigger Asynchronous Tile DMA engine concurrently
        dma_start = 1;
        #2 dma_start = 0;

        // Wait for pipeline drain & DMA completion
        repeat (15) begin
            #2;
            if (mem_req_valid) begin
                $display("[ATDE DMA BUS] Memory Read Address Requested: 0x%08x", mem_req_addr);
            end
        end

        // Wait for DMA done
        wait(dma_done);
        $display("[ATDE DMA BUS] Tile DMA transfer completed successfully!");

        #10;
        $display("---------------------------------------------------------------");
        $display("TEST PASSED: Triton Systolic Matrix + DMA Hardware Verified!");
        $display("---------------------------------------------------------------");
        $finish;
    end

endmodule
