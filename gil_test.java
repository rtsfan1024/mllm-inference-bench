class GilTest {
    // 空循环20亿次
    static void work() { for (long i = 0; i < 2_000_000_000L; i++); }

    public static void main(String[] args) throws Exception {
        // 串行跑两次
        long start = System.currentTimeMillis();
        work(); work();
        System.out.println("串行耗时: " + (System.currentTimeMillis() - start) + "ms");

        // 并行跑两次
        start = System.currentTimeMillis();
        Thread t1 = new Thread(GilTest::work);
        Thread t2 = new Thread(GilTest::work);
        t1.start(); t2.start(); // 同时开跑
        t1.join(); t2.join();   // 等待都跑完
        System.out.println("并行耗时: " + (System.currentTimeMillis() - start) + "ms");
    }
}



